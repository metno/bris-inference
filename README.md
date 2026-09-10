# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on
the [Anemoi framework](https://github.com/ecmwf/anemoi-training).

## Features

- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation
- Ensembles
- Verification observation files from analysis datasets (`bris-obs`)

## Documentation

See [Wiki](https://github.com/metno/bris-inference/wiki)

## Verifying against an analysis dataset

`bris-obs --config obs.yaml` extracts a subset of points from an anemoi-datasets zarr
store over a period and writes one Verif observation file per variable. Point the
`verif` observation source of the `verif` output at these files to verify forecasts
against the analysis. The config takes an open_dataset-style `dataset` recipe (a
path, `join` of stores on the same grid, `area`, `frequency`, `every_loc` or `spacing`) and a list of
`verif` outputs (`filename`, `variable`, `units`); see `config/verif_obs_template.yaml`.
The stores are read directly, so other open_dataset operations are not supported.

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
