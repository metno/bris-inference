from collections import defaultdict
import numpy as np
import xarray as xr


from bris.output import Output
from bris.predict_metadata import PredictMetadata
from bris import cf
from bris import utils
from bris.outputs.intermediate import Intermediate


class GriddedNetcdf(Output):
    """Write predictions to MET-Norway's way of structuring NetCDF files.

    Since ensemble is done data-parallel, we do not have all members available when writing the
    files. If we are producing a single deterministic run, then we can directly write data to file
    as soon as we get it. Otherwise write the data to disk in an intermediate format and then merge
    files on finalize. This comes at a penalty since the data is written to disk twice.
    """
    def __init__(self, predict_metadata: PredictMetadata, filename_pattern: str):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
        """
        super().__init__(predict_metadata)
        self.filename_pattern = filename_pattern

        self.intermediate = None
        if self.pm.num_members > 1:
            workdir = "test/"
            self.intermediate = Intermediate(predict_metadata, workdir)

    def _add_forecast(self, forecast_reference_time: int, ensemble_member: int, pred: np.array):
        if self.pm.num_members > 1:
            # Cache data with intermediate
            self.intermediate.add_forecast(forecast_reference_time, ensemble_member, pred)
            return
        else:
            assert ensemble_member == 0

            filename = self.get_filename(forecast_reference_time)
            # Add ensemble dimension to the last
            self.write(filename, forecast_reference_time, pred[..., None])

    def get_filename(self, forecast_reference_time):
        return utils.expand_time_tokens(self.filename_pattern, forecast_reference_time)

    def write(self, filename: str, forecast_reference_time: int, pred: np.array):
        """Write prediction to NetCDF
        Args:
            forecast_reference_time: Time that this forecast is for
            pred: 4D numpy array with dimensions (leadtimes, points, variables, members)
        """

        coords = dict()

        # TODO: Seconds or hours for leadtimes?
        times = [forecast_reference_time + lt for lt in self.pm.leadtimes]
        coords["time"] = np.array(times).astype(np.double)

        # TODO: Deal with non-regular grids
        x = np.arange(self.pm.field_shape[1]).astype(np.float32)
        y = np.arange(self.pm.field_shape[0]).astype(np.float32)
        coords["x"] = x
        coords["y"] = y

        if self.pm.num_members > 1:
            coords["ensemble_member"] = range(self.pm.num_members)

        dims_to_add, ncname_to_level_dim, variable_to_ncname_and_index = cf.get_list(self.pm.variables)

        attrs = dict()
        # Add dimensions
        for dimname, (level_type, levels) in dims_to_add.items():
            coords[dimname] = np.array(levels).astype(np.float32)
            attrs[dimname] = cf.get_attributes_from_leveltype(level_type)

        for k,v in self._get_attrs().items():
            if k in coords:
                attrs[k] = v
        self.ds = xr.Dataset(coords=coords)

        # Add attributes of coordinates
        for var, var_attrs in attrs.items():
            self.ds[var].attrs = var_attrs

        # Set up other coordinate variables
        self.ds["forecast_reference_time"] = ([], float(forecast_reference_time))
        print(self.pm.lats.shape, self.pm.field_shape, len(y), len(x))
        lats = np.reshape(self.pm.lats, self.pm.field_shape).astype(np.double)
        lons = np.reshape(self.pm.lons, self.pm.field_shape).astype(np.double)
        self.ds["latitude"] = (["y", "x"], lats)
        self.ds["longitude"] = (["y", "x"], lons)

        for ncname in ["forecast_reference_time", "time", "latitude", "longitude", "x", "y"]:
            self.ds[ncname].attrs = cf.get_attributes_from_ncname(ncname)

        # Set up all prediction variables
        for variable_index, variable in enumerate(self.pm.variables):
            ncname, level_index = variable_to_ncname_and_index[variable]

            if ncname not in self.ds:
                dim_name = ncname_to_level_dim[ncname]
                if self.pm.num_members > 1:
                    dims = ["time", dim_name, "ensemble_member", "y", "x"]
                else:
                    dims = ["time", dim_name, "y", "x"]
                shape = [len(times), len(self.ds[dim_name]), len(y), len(x)]
                ar = np.zeros(shape, np.float32)
                self.ds[ncname] = (dims, ar)
            if self.pm.num_members > 1:
                shape = [len(times), len(y), len(x), self.pm.num_members]
            else:
                shape = [len(times), len(y), len(x)]
            ar = np.reshape(pred[..., variable_index, :], shape)

            if self.pm.num_members > 1:
                # Move ensemble dimension into the middle position
                ar = np.moveaxes(ar, [0, 1, 2, 3, 4], [0, 1, 3, 4, 2])

            self.ds[ncname][:, level_index, ...] = ar
            # TODO:
            self.ds[ncname].attrs = {"units": "unknown"}

        """
        # Assign data
        for v, variable in enumerate(self.pm.variables):
            ncname, index = variable_to_ncname_and_index[variable]
            self.ds[ncname][:, index, ...] = pred[..., v]
        """

        self.ds.to_netcdf(filename)

    def finalize(self):
        if self.intermediate is not None:
            # Load data from the intermediate and write to disk
            forecast_reference_times = self.intermediate.get_forecast_reference_times()
            for forecast_reference_time in forecast_reference_times:
                # Arange all ensemble members
                pred = np.zeros(self.pm.shape + [self.pm.num_members], np.float32)
                for m in range(self.pm.num_members):
                    curr = self.intermediate.get_forecast(forecast_reference_time, m)
                    if curr is not None:
                        pred[..., m] = curr

                filename = self.get_filename(forecast_reference_time)
                self.write(filename, forecast_reference_times, pred)

    def _get_attrs(self):
        attrs = dict()
        attrs["forecast_reference_time"] = {
                "units": "seconds since 1970-01-01 00:00:00 +00:00",
                "standard_name": "forecast_reference_time"
            }
        attrs["time"] = {
                "units": "seconds since 1970-01-01 00:00:00 +00:00",
                "standard_name": "time"
            }
        attrs["latitude"] = {
                "units": "degrees_north",
                "standard_name": "latitude"
            }
        attrs["longitude"] = {
                "units": "degrees_east",
                "standard_name": "longitude"
            }
        attrs["x"] = {
                "units": "m",
                "standard_name": "projection_x_coordinate"
                }
        attrs["y"] = {
                "units": "m",
                "standard_name": "projection_y_coordinate"
                }
        return attrs

"""
class VariableList:
    def __init__(self, anemoi_names):
        self.anemoi_names = anemoi_names

    def get_dimension_definitions(self):
        pass

    def get_variable_definitions(self):
        pass

    def get_ncname(anemoi_name):
    pass
"""
