import contextlib
import glob
import os

from bris import zarrstore
from bris.outputs.zarr import Zarr
from bris.predict_metadata import PredictMetadata


class Netcdf(Zarr):
    """Write predictions to NetCDF, using CF-standards and local conventions

    Since ensemble is done data-parallel, we do not have all members available when writing the
    files. The predictions are therefore written to a Zarr store in the work directory, which
    every member can write into at the same time (see bris.outputs.zarr), and the store is
    converted to NetCDF when it is complete (see bris.convert): as soon as it is written for a
    single deterministic run, otherwise on finalize. This comes at a penalty since the data is
    written to disk twice.

    This output can write three types of outputs (see bris.outputs.layout):
    1) Gridded regional projected data. This requires field_shape to be set in predict_metadata
    2) Irregular grids interpolated to a lat/lon grid. Use interp_res.
    3) Irregular grids. This means the output only has one location dimension.
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        variables=None,
        interp_res=None,
        latrange=None,
        lonrange=None,
        extra_variables=None,
        accumulated_variables=None,
        proj4_str=None,
        domain_name=None,
        mask_file: str = "",
        mask_field=None,
        global_attributes=None,
        remove_intermediate=True,
        compression=False,
    ):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            variables: If None, predict all variables
            global_attributes (dict): Write these global attributes in the output file
            remove_intermediate (bool): Remove the Zarr store when the file has been written
            compression (bool): If true, write compressed output files
        """
        super().__init__(
            predict_metadata,
            workdir,
            filename_pattern,
            variables=variables,
            interp_res=interp_res,
            latrange=latrange,
            lonrange=lonrange,
            extra_variables=extra_variables,
            accumulated_variables=accumulated_variables,
            proj4_str=proj4_str,
            domain_name=domain_name,
            mask_file=mask_file,
            mask_field=mask_field,
            global_attributes=global_attributes,
            # One field per chunk, as in the file
            chunks={"time": 1},
            remove_after_convert=remove_intermediate,
        )
        self.netcdf_compression = compression

    def get_store(self, forecast_reference_time: int) -> str:
        return f"{self.workdir}/{forecast_reference_time:.0f}.zarr"

    def get_stores(self) -> list[str]:
        return sorted(
            path
            for path in glob.glob(f"{glob.escape(self.workdir)}/*.zarr")
            if zarrstore.is_from_this_run(path)
        )

    @property
    def has_targets(self) -> bool:
        return True

    def get_targets(self, forecast_reference_time: int) -> list[dict]:
        return [
            {
                "format": "netcdf",
                "filename": self.get_filename(forecast_reference_time),
                "compression": bool(self.netcdf_compression),
            }
        ]

    def _cleanup(self) -> None:
        """Remove the work directory, unless it still holds stores"""
        with contextlib.suppress(OSError):
            os.rmdir(self.workdir)
