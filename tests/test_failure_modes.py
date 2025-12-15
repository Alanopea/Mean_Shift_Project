import pytest
from mean_shift_project.mean_shift import MeanShift

def test_predict_before_fit_raises_error():
    ms = MeanShift()
    with pytest.raises(Exception):
        ms.predict([[0.0, 0.0]])
