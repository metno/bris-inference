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

## Install

### Locally for development

    python3 -m venv venv && source venv/bin/activate
    pip install -e .

## How to run Bris

    bris --config config.yaml

## How to run tests

    pip install ".[tests]"
    tox

List all tests, and run a single one:

    $ tox -a
    py310
    py311
    py312
    ruffformat
    ruffcheck
    typing
    prospector
    bandit

    $ tox -e bandit

Only py31x tests are expected to run without error so far.

## Code borrowed from Anemoi project

- bris/ddp_strategy.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/distributed/strategy.py>
- bris/grid_indices.py is based on <https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/data/grid_indices.py>
- bris/data/data{set,module}.py is somewhat based on <https://github.com/ecmwf/anemoi-core/tree/main/training/src/anemoi/training/data>
