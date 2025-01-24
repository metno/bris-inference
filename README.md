# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on the Anemoi framework
(https://github.com/ecmwf/anemoi-training).

# Features
- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation

# Requirements

- udunits2 library. On ubuntu available as `libudunits2-0`

# How to run Bris

```python

bris --config config.yaml
```
