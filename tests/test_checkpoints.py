import os

from anemoi.utils.config import DotDict

import bris.checkpoint


def test_metadata():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/checkpoint.ckpt"
    checkpoint = bris.checkpoint.Checkpoint(path=filename)

    n2i = (
        {
            "10u": 0,
            "10v": 1,
            "2d": 2,
            "2t": 3,
            "cos_julian_day": 4,
            "cos_latitude": 5,
            "cos_local_time": 6,
            "cos_longitude": 7,
            "cp": 8,
            "insolation": 9,
            "lsm": 10,
            "msl": 11,
            "q_100": 12,
            "q_1000": 13,
            "q_150": 14,
            "q_200": 15,
            "q_250": 16,
            "q_300": 17,
            "q_400": 18,
            "q_50": 19,
            "q_500": 20,
            "q_600": 21,
            "q_700": 22,
            "q_850": 23,
            "q_925": 24,
            "sdor": 25,
            "sin_julian_day": 26,
            "sin_latitude": 27,
            "sin_local_time": 28,
            "sin_longitude": 29,
            "skt": 30,
            "slor": 31,
            "sp": 32,
            "t_100": 33,
            "t_1000": 34,
            "t_150": 35,
            "t_200": 36,
            "t_250": 37,
            "t_300": 38,
            "t_400": 39,
            "t_50": 40,
            "t_500": 41,
            "t_600": 42,
            "t_700": 43,
            "t_850": 44,
            "t_925": 45,
            "tcw": 46,
            "tp": 47,
            "u_100": 48,
            "u_1000": 49,
            "u_150": 50,
            "u_200": 51,
            "u_250": 52,
            "u_300": 53,
            "u_400": 54,
            "u_50": 55,
            "u_500": 56,
            "u_600": 57,
            "u_700": 58,
            "u_850": 59,
            "u_925": 60,
            "v_100": 61,
            "v_1000": 62,
            "v_150": 63,
            "v_200": 64,
            "v_250": 65,
            "v_300": 66,
            "v_400": 67,
            "v_50": 68,
            "v_500": 69,
            "v_600": 70,
            "v_700": 71,
            "v_850": 72,
            "v_925": 73,
            "w_100": 74,
            "w_1000": 75,
            "w_150": 76,
            "w_200": 77,
            "w_250": 78,
            "w_300": 79,
            "w_400": 80,
            "w_50": 81,
            "w_500": 82,
            "w_600": 83,
            "w_700": 84,
            "w_850": 85,
            "w_925": 86,
            "z": 87,
            "z_100": 88,
            "z_1000": 89,
            "z_150": 90,
            "z_200": 91,
            "z_250": 92,
            "z_300": 93,
            "z_400": 94,
            "z_50": 95,
            "z_500": 96,
            "z_600": 97,
            "z_700": 98,
            "z_850": 99,
            "z_925": 100,
        },
    )

    assert checkpoint.metadata.version == "1.0", "version is not 1.0"
    assert checkpoint.metadata.run_id == "775d1ad8-4457-4268-a430-3df91cc55603", (
        "run_id seems wrong"
    )
    assert isinstance(checkpoint.metadata.config, DotDict), "config is not DotDict"
    assert isinstance(checkpoint.metadata.dataset, DotDict), "dataset is not DotDict"
    assert isinstance(checkpoint.metadata.data_indices, DotDict), (
        "data_indices is not DotDict"
    )

    assert checkpoint.config is not None, "config is None"
    assert checkpoint.graph is not None, "graph is None"

    assert checkpoint.name_to_index == n2i, "name_to_index is not correct"


if __name__ == "__main__":
    test_metadata()
