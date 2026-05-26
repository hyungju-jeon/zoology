import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment, replace_zero_with_random_non_keys


class ChainMQARConfig(DataSegmentConfig):
    name: str = "chain_mqar"
    power_a: float = 0.01
    num_kv_pairs: int = 8
    num_updates: int = 4  # How many reassignment events to emit
    multiple_update: bool = False
    random_non_queries: bool = True
    shuffle_kv_set: bool = False
    include_slices: bool = True

    def build(self, seed: int) -> DataSegment:
        return chain_mqar(**self.model_dump(), seed=seed)


def chain_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    shuffle_kv_set: bool = False,
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """
    Generate chained MQAR data from adjacent one-hop edge writes.

    Each chain is ``A_i -> B_i -> C_i``.  The context contains shuffled adjacent
    second-hop edge pairs ``B_i C_i`` followed by shuffled adjacent first-hop
    edge pairs ``A_i B_i``; queries contain only ``A_i`` and the label is
    ``C_i``.  This preserves the local key/value binding pressure of standard
    MQAR while requiring two-hop recall from an anti-causal write order.
    """
    vocab_size = int(vocab_size)
    num_examples = int(num_examples)
    input_seq_len = int(input_seq_len)
    num_passes = int(num_passes)
    num_kv_pairs = int(num_kv_pairs)
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    if num_passes != 1:
        raise ValueError("two-hop MQAR currently supports only `num_passes=1`.")
    if 3 * num_kv_pairs > vocab_size - 1:
        raise ValueError(
            "two-hop MQAR needs at least three disjoint token sets of size "
            f"`num_kv_pairs`; got vocab_size={vocab_size}, num_kv_pairs={num_kv_pairs}."
        )
    assert (
        6 * num_kv_pairs <= input_seq_len
    ), f"Need at least 6 tokens per two-hop chain. Got {input_seq_len} for {num_kv_pairs} chains."

    import numpy as np

    np.random.seed(int(seed))
    context_size = num_kv_pairs * 4

    if shuffle_kv_set:
        token_choices = np.arange(1, vocab_size)
        shuffled_choices = (
            np.stack([np.random.permutation(token_choices) for _ in range(num_examples)], axis=0)
            if num_examples > 0
            else np.empty((0, vocab_size - 1), dtype=np.int64)
        )
        roots = shuffled_choices[:, :num_kv_pairs]
        bridges = shuffled_choices[:, num_kv_pairs : 2 * num_kv_pairs]
        finals = shuffled_choices[:, 2 * num_kv_pairs : 3 * num_kv_pairs]
    else:
        third = vocab_size // 3
        root_choices = np.arange(1, third)
        bridge_choices = np.arange(third, 2 * third)
        final_choices = np.arange(2 * third, vocab_size)
        if min(root_choices.size, bridge_choices.size, final_choices.size) < num_kv_pairs:
            raise ValueError(
                "two-hop MQAR disjoint vocabulary thirds are too small for "
                f"num_kv_pairs={num_kv_pairs}."
            )
        if num_examples > 0:
            roots = np.apply_along_axis(
                np.random.choice,
                1,
                np.tile(root_choices, (num_examples, 1)),
                replace=False,
                size=num_kv_pairs,
            )
            bridges = np.apply_along_axis(
                np.random.choice,
                1,
                np.tile(bridge_choices, (num_examples, 1)),
                replace=False,
                size=num_kv_pairs,
            )
            finals = np.apply_along_axis(
                np.random.choice,
                1,
                np.tile(final_choices, (num_examples, 1)),
                replace=False,
                size=num_kv_pairs,
            )
        else:
            roots = np.empty((0, num_kv_pairs), dtype=np.int64)
            bridges = np.empty((0, num_kv_pairs), dtype=np.int64)
            finals = np.empty((0, num_kv_pairs), dtype=np.int64)

    first_hop_edges = np.stack([roots, bridges], axis=-1)
    second_hop_edges = np.stack([bridges, finals], axis=-1)
    edge_pairs = np.zeros((num_examples, num_kv_pairs * 2, 2), dtype=np.int64)
    for example_idx in range(num_examples):
        first_order = np.random.permutation(num_kv_pairs)
        second_order = np.random.permutation(num_kv_pairs)
        edge_pairs[example_idx, :num_kv_pairs] = second_hop_edges[example_idx, second_order]
        edge_pairs[example_idx, num_kv_pairs:] = first_hop_edges[example_idx, first_order]
    kvs = edge_pairs.reshape(num_examples, context_size)

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    if num_examples > 0:
        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(
            np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
        )
    else:
        gaps = np.empty((0, num_kv_pairs), dtype=np.int64)

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=roots, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=finals, axis=1)

    inputs_np = examples[:, :-1].copy()
    if random_non_queries:
        inputs_np = replace_zero_with_random_non_keys(
            inputs_np,
            roots,
            vocab_size=vocab_size,
            seed=int(seed) + 1_337,
            min_token=1,
        )

    inputs, labels = torch.tensor(inputs_np), torch.tensor(labels[:, 1:])
    return DataSegment(
        inputs,
        labels,
        slices=(
            {
                "num_kv_pairs": num_kv_pairs,
                "input_seq_len": input_seq_len,
                "two_hop": True,
                "shuffle_kv_set": bool(shuffle_kv_set),
            }
            if include_slices
            else {}
        ),
    )
