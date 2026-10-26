import numpy as np

import numpy as np

from dual_track_opd.scoped_runtime.collate import (
    as_sample_object_array,
    normalize_non_tensor_batch,
)


def test_equal_length_sample_lists_stay_one_dimensional():
    value = as_sample_object_array([[1, 2, 3], [4, 5, 6]])

    assert value.shape == (2,)
    assert value.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_mixed_sample_objects_preserve_the_sample_axis():
    value = as_sample_object_array([[1, 2], "answer", {"images": []}])

    assert value.shape == (3,)
    assert value[0] == [1, 2]
    assert value[1] == "answer"
    assert value[2] == {"images": []}


def test_numpy_rows_are_normalized_back_to_sample_objects():
    nested = np.array([[1, 2], [3, 4]], dtype=object)
    value = as_sample_object_array(nested)

    assert value.shape == (2,)
    assert value[0] == [1, 2]
    assert value[1] == [3, 4]


def test_normalize_non_tensor_batch_flattens_rectangular_object_arrays():
    mapping = {
        "raw_prompt_ids": np.array([[1, 2], [3, 4]], dtype=object),
        "ground_truth": np.array(["a", "b"], dtype=object),
        "numeric": np.array([[1, 2], [3, 4]], dtype=np.int64),
    }

    result = normalize_non_tensor_batch(mapping)

    assert result["raw_prompt_ids"].shape == (2,)
    assert result["raw_prompt_ids"].tolist() == [[1, 2], [3, 4]]
    assert result["ground_truth"].shape == (2,)
    assert result["numeric"].shape == (2, 2)
