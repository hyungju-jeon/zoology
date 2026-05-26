import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment, replace_zero_with_random_non_keys


class BlockMQARConfig(DataSegmentConfig):
    name: str = "block_mqar"
    power_a: float = 0.01
    num_kv_pairs: int = 8
    num_updates: int = 4  # How many reassignment events to emit
    multiple_update: bool = False
    random_non_queries: bool = True
    shuffle_kv_set: bool = False
    include_slices: bool = True

    def build(self, seed: int) -> DataSegment:
        return block_mqar(**self.model_dump(), seed=seed)


def block_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    shuffle_kv_set: bool = False,
    kv_block_size: int = 2,
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """
    Generate MQAR data with each context block laid out as keys followed by values.

    With ``kv_block_size=2``, the context is:
        K1 K2 V1 V2  K3 K4 V3 V4  ...

    This keeps the usual MQAR query/label structure, but the value for each key is
    more than one step after the key in the context.
    """
    vocab_size = int(vocab_size)
    num_examples = int(num_examples)
    input_seq_len = int(input_seq_len)
    num_passes = int(num_passes)
    kv_block_size = int(kv_block_size)
    num_kv_pairs = int(num_kv_pairs)
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    if kv_block_size <= 1:
        raise ValueError("`kv_block_size` must be > 1 for separated MQAR.")
    if kv_block_size > num_kv_pairs:
        raise ValueError(
            "`kv_block_size` must be <= `num_kv_pairs` so each context block is complete."
        )
    if num_kv_pairs % kv_block_size != 0:
        raise ValueError(
            "`num_kv_pairs` must be divisible by `kv_block_size` so every key/value "
            "pair has the configured separation."
        )

    np_seed = int(seed)
    import numpy as np

    np.random.seed(np_seed)

    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)
    if shuffle_kv_set:
        token_choices = np.arange(1, vocab_size)
        shuffled_choices = (
            np.stack([np.random.permutation(token_choices) for _ in range(num_examples)], axis=0)
            if num_examples > 0
            else np.empty((0, vocab_size - 1), dtype=np.int64)
        )
        keys_unshuffled = shuffled_choices[:, : key_choices.shape[0]]
        values_unshuffled = shuffled_choices[
            :, key_choices.shape[0] : key_choices.shape[0] + value_choices.shape[0]
        ]
    else:
        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        values_unshuffled = np.tile(value_choices, (num_examples, 1))

    if num_examples > 0:
        keys = np.apply_along_axis(
            np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
        )
        values = np.apply_along_axis(
            np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
        )
    else:
        keys = np.empty((0, num_kv_pairs), dtype=np.int64)
        values = np.empty((0, num_kv_pairs), dtype=np.int64)

    one_pass_context = np.zeros((num_examples, num_kv_pairs * 2), dtype=np.int64)
    block_width = kv_block_size * 2
    for pair_start in range(0, num_kv_pairs, kv_block_size):
        pair_end = pair_start + kv_block_size
        block_start = (pair_start // kv_block_size) * block_width
        one_pass_context[:, block_start : block_start + kv_block_size] = keys[
            :, pair_start:pair_end
        ]
        one_pass_context[:, block_start + kv_block_size : block_start + block_width] = values[
            :, pair_start:pair_end
        ]

    kvs = np.tile(one_pass_context, (1, num_passes))

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
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs_np = examples[:, :-1].copy()
    if random_non_queries:
        inputs_np = replace_zero_with_random_non_keys(
            inputs_np,
            keys,
            vocab_size=vocab_size,
            seed=np_seed + 1_337,
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
                "num_passes": num_passes,
                "kv_block_size": kv_block_size,
                "shuffle_kv_set": bool(shuffle_kv_set),
            }
            if include_slices
            else {}
        ),
    )
