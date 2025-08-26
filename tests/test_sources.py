import os

from bris import sources


def test_instantiate():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    args = {"filename": filename, "obs_variable": "obs"}
    _ = sources.instantiate("verif", args)

    args = {"filename": filename}
    _ = sources.instantiate("verif", args)


if __name__ == "__main__":
    test_instantiate()
