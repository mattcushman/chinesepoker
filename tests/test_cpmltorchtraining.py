# pytest module to test TorchAgent/CPMLTorchTraining.py
import pytest
from TorchAgent import CPMLTorchTraining

def test_main():
    args = [
        '--seed', '1',
    ]

    CPMLTorchTraining.main(args)
    assert True



