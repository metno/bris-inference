import os

import bris.utils


def test_1():
    filenames = ["working_example.yaml"]
    for filename in filenames:
        full_filename = (
            os.path.dirname(os.path.abspath(__file__)) + "/../config/" + filename
        )
        bris.utils.validate(full_filename)


if __name__ == "__main__":
    test_1()
