import os
import numpy as np

import bris.routes


def test_get():
    config = list()
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    config += [{
        "decoder_index": 0,
        "domain": 0,
        "outputs": {
            "verif": {
                "filename": "nordic/2t/%R.nc",
                "variable": "2t",
                "units": "C",
                "thresholds": [0, 10, 20],
                "quantile_levels": [0.1, 0.9],
                "obs_sources": [{"verif": {"filename": filename}}],
            }
        },
    }]
    data_module = None
    run_name = "legendary_gnome"
    workdir = "testdir"

    routes = bris.routes.get(config, data_module, run_name, workdir)
    print(routes)


if __name__ == "__main__":
    test_get()
