# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on
the [Anemoi framework](https://github.com/ecmwf/anemoi-training).

## Features

- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation

## Documentation

See [Wiki](https://github.com/metno/bris-inference/wiki)

## Requirements

- udunits2 library. On ubuntu available as `libudunits2-0`
- Running on ARM requires some workarounds for now: https://github.com/metno/bris-inference/issues/85

## Install

### Locally for development

    python3 -m venv venv && source venv/bin/activate
    pip install -e .

### Via docker, if you are from Met.no

See [Dockerfile](https://gitlab.met.no/yrop/bris-cicd/-/blob/main/Dockerfile?ref_type=heads)

## How to run tests

    pip install ".[tests]"
    tox list
    tox

When pushing to github, default tests will be run automatically and must succeed.
Read more about [Tests](https://github.com/metno/bris-inference/wiki/Tests)
in the wiki.

## Code borrowed from Anemoi project

- bris/ddp_strategy.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/distributed/strategy.py>
- bris/grid_indices.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/data/grid_indices.py>
- bris/data/data{set,module}.py is somewhat based on <https://github.com/ecmwf/anemoi-core/tree/main/training/src/anemoi/training/data>
