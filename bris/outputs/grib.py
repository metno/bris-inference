import contextlib
import os

from bris import utils
from bris.outputs.zarr import Zarr, _plain
from bris.predict_metadata import PredictMetadata


class Grib(Zarr):
    """Write predictions to Grib, using CF-standards and local conventions

    The predictions are written to a Zarr store in the work directory (see bris.outputs.zarr),
    with the same variables, units and grid as the NetCDF output, and the store is converted to
    Grib when it is complete (see bris.convert).
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        variables=None,
        interp_res=None,
        grib_keys: dict = None,
        proj4_str=None,
        domain_name=None,
        extra_variables=None,
        remove_intermediate: bool = True,
        convert_at_end=True,
    ):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
            variables: If None, predict all variables
            grib_keys (dict): Set these keys in every message
            remove_intermediate (bool): Remove the Zarr store when the file has been written
            convert_at_end (bool): Write the files when the predictions end. Otherwise the Zarr
                stores are left in the work directory for bris-convert.
        """
        if interp_res is not None:
            utils.LOGGER.warning("grib: interp_res is not supported and is ignored")
        super().__init__(
            predict_metadata,
            workdir,
            filename_pattern,
            variables=variables,
            extra_variables=extra_variables,
            proj4_str=proj4_str,
            domain_name=domain_name,
            # One field per chunk, as the messages in the file
            chunks={"time": 1},
            remove_after_convert=remove_intermediate,
            convert_at_end=convert_at_end,
        )
        self.grib_keys = _plain(grib_keys or {})

    def get_store(self, forecast_reference_time: int) -> str:
        return f"{self.workdir}/{forecast_reference_time:.0f}.zarr"

    @property
    def has_targets(self) -> bool:
        return True

    def get_targets(self, forecast_reference_time: int) -> list[dict]:
        return [
            {
                "format": "grib",
                "filename": self.get_filename(forecast_reference_time),
                "grib_keys": self.grib_keys,
            }
        ]

    def _cleanup(self) -> None:
        """Remove the work directory, unless it still holds stores"""
        with contextlib.suppress(OSError):
            os.rmdir(self.workdir)
