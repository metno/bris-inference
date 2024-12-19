import os

import numpy as np
from bris import sources
from bris.sources import Verif


def test_instantiate():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    args = {"filename": filename}

    obs_source = sources.instantiate("verif", args)


if __name__ == "__main__":
    test_instantiate()
