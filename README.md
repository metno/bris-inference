# Bris inference

This is a package to run MET Norway's data-driven model Bris, which is based on
the [Anemoi framework](https://github.com/ecmwf/anemoi-training).

## Features

- Model and data-parallel inference
- Multi encoder/decoder
- Time interpolation
- Ensembles

## Documentation

See [Wiki](https://github.com/metno/bris-inference/wiki)

### Polygon output

Install the optional GIS dependencies with `pip install 'bris[polygons]'` (or
`pip install -e '.[polygons]'` for a checkout). Add a `polygon` output to a routing
entry:

```yaml
outputs:
  - polygon:
      filename_pattern: /path/to/forecast_%Y%m%dT%HZ.zarr
      polygon_files:
        - /path/to/catchments_shop.feather
      format: zarr                 # netcdf (default) or zarr
      spatial_chunk_size: 256      # maximum along each spatial dimension
      compression: true           # default
      variables: [2t, tp]
      accumulated_variables: [tp]
```

To write NetCDF, set `format: netcdf` and use a `.nc` filename. The format is
explicit, rather than inferred from the filename. Feather, GeoParquet, GeoJSON,
shapefiles and other GeoPandas-supported vector files are accepted. Extract
Feather files from ZIP archives before configuring their paths. Every file must
declare its coordinate reference system; forecast coordinates are transformed to
that CRS automatically, including UTM coordinates in the example catchment files.

All features from all supplied files are combined as a union. Grid points are
selected by their centres, including polygon boundaries; holes are excluded and
values are not averaged over polygons. Regular grids keep their original extent,
coordinates and dimensions, with NaNs outside the polygons. Irregular grids store
only selected locations, retaining their original location indices. For
`interp_res` or `mask_file` outputs, selection uses the final reconstructed grid.
An error is raised if no output points are covered. Other NetCDF options, such as
derived variables, projections, attributes and ensembles, are also supported.

Both formats use lossless compression by default: NetCDF uses zlib, while Zarr v2
uses Blosc/Zstd with bitshuffle and omits all-NaN prediction chunks. Time, level
and ensemble chunks each contain one element. Smaller spatial chunks can save
space for scattered polygons but create more chunks; the best size depends on
the grid resolution and polygon coverage. Zarr suits sparse gridded analysis;
NetCDF provides a single file compatible with existing NetCDF workflows. Ensemble
forecasts use the existing temporary per-member cache before final writing.

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
