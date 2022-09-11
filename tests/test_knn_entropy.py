import pytest
import numpy as np
import numpy.testing

import copent

@pytest.mark.parametrize("dtype, entropy", [
    ('euclidean', 5.68),
    ('chebychev', 5.73),
])
def test_knn_entropy(dtype, entropy):
    arr = np.array([[ 1,  2], [ 2,  5], [10, 10]])

    res = copent.entknn(arr, dtype=dtype, k=1)
    assert res == pytest.approx(entropy, abs=0.01)
