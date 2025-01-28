# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on the [Anemoi framework](https://github.com/ecmwf/anemoi-training).

# Features
- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation

# Documentation

See [Wiki](https://github.com/metno/bris-inference/wiki)

# Requirements

- udunits2 library. On ubuntu available as `libudunits2-0`

# Install

## Locally for development

    $ python3 -m venv venv && source venv/bin/activate
    $ pip install -e .

# How to run Bris

    $ bris --config config.yaml

# How to run tests

    $ tox
