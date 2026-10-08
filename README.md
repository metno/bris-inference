# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on
the [Anemoi framework](https://github.com/ecmwf/anemoi-training).

## Features

- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation
- Ensembles
- Multisource nowcast input preparation and packaged inference configuration
- Calibrated lightning postprocessing and evolution plots
- Precipitation member-evolution plots and GIF previews

## Operational nowcasting

The `nowcasting` branch packages the Python side of the operational pipeline:

- `bris-nowcast-helper`: resolves cycles, loads the launcher config, renders the
  four-source BRIS config, and validates it against the checkpoint contract.
- `bris-nowcast-inputs`: prepares raw Nordic radar, latest-valid-time MEPS,
  Netatmo, and lightning inputs through Weathermart.
- `bris-lightning-postprocess`: converts decoder output into the fixed-grid,
  calibrated +5-to-+30-minute lightning product and operational plot.
- `bris-lightning-evolution`: creates the selected-window lightning product and
  evolution plot.
- `plot_nowcast`: creates the precipitation member-evolution plot and GIF.

The inference template, Netatmo station list, and fixed lightning-grid
definition are package data under `bris/schema`. The current checkpoint uses
the four sources above and has no Rainbow dataset.

## Plotting a nowcast

The branch installs one precipitation plotting command:

```bash
plot_nowcast precipitation.nc precipitation_member_evolution.png \
  --lead-minutes 5 10 20 60 90 120 \
  --animation precipitation_members_median.gif
```

Unless four member IDs are supplied with `--members`, the plotter selects the
four ensemble members with the greatest spatial diversity. The static plot
shows those members at six lead times; the GIF shows the same four members
around the ensemble median.

## Documentation

See [Wiki](https://github.com/metno/bris-inference/wiki)

## Requirements

- Running on ARM/MacOS requires some workarounds for now: https://github.com/metno/bris-inference/issues/85

## Install

### Locally for development

    python3 -m venv venv && source venv/bin/activate
    pip install -e .

### From PIP

    pip install bris

### Via docker, if you are Met.no employee

See [Dockerfile](https://gitlab.met.no/yrop/bris-cicd/-/blob/main/Dockerfile?ref_type=heads)

## How to run tests

    pip install -e '.[dev]'
    tox

When pushing to github, default tests will be run automatically and must succeed.
Read more about [Tests](https://github.com/metno/bris-inference/wiki/Tests)
in the wiki.

## Code borrowed from Anemoi project

- bris/ddp_strategy.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/distributed/strategy.py>
- bris/grid_indices.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/data/grid_indices.py>
- bris/data/data{set,module}.py is somewhat based on <https://github.com/ecmwf/anemoi-core/tree/main/training/src/anemoi/training/data>
