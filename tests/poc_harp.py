import os
import tempfile
import numpy as np
import pytest
from bris.outputs import Harp
from bris.predict_metadata import PredictMetadata
from bris.sources import NetCDF as NetCDFInput
import pyarrow.parquet as pq
import csv
import xarray as xr
from geopy.distance import geodesic
from scipy.spatial import Delaunay
import gridpp
import netCDF4
import pyproj
import pyarrow as pa
from pyarrow import dataset as pa_dataset
import time

obsfile=os.path.dirname(os.path.abspath(__file__)) + '/files/observations_for_harp.nc'
fcstfile=os.path.dirname(os.path.abspath(__file__)) + '/files/meps_pred_20230615T06Z_small.nc'
parameters=["T2m"]
outdir=f"/fctable/{parameters[0]}/"


def find_closest_value(ds, lats, lons, unixtime: np.datetime64, variable: str):
    """ Find variable value closest to coords at given time"""
    # projection = ds.variables["projection_lambert"].proj4
    # proj = pyproj.Proj(projection)

    # for meps:
    proj = pyproj.Proj('+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs')
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]
    times = ds.variables["time"][:]
    t = np.argmin(np.abs(times - unixtime))

    X, Y = proj(lons, lats)
    Ix = np.argmin(np.abs(x - X))
    Iy = np.argmin(np.abs(y - Y))

    return ds.variables[variable][t, 0, Iy, Ix]


def alt_correction(temperature: float, altitude: int) -> float:
    """ Ajust temperature for altitude http://walter.bislins.ch/bloge/index.asp?page=Barometric+Formula """
    elev_gradient = -0.0065
    return temperature+(altitude*elev_gradient)


def test_export_to_harp():
    # Create a prediction
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(variables, lats, lons, altitudes, leadtimes, num_members, field_shape)

    forecast_ds = netCDF4.Dataset(filename=fcstfile, mode='r')
    observations_ds = xr.open_dataset(obsfile)

    # pyarrow arrays
    locations = []
    lats = []
    lons = []
    alts = []
    time_list = []
    valid_dttm = []
    obs = []
    fcst = []

    # Loop station_ids
    for loc_idx in range(observations_ds.sizes['location']):
        location = observations_ds.variables['location'][loc_idx]

        # Loop times
        for time_idx in range(observations_ds.sizes['time']):
            unixtime = observations_ds.variables['time'][time_idx].values.item()
            observation = observations_ds.variables['obs'][time_idx, loc_idx]
            if time_idx == 0:
                print(f"Location (id, lat, lon, alt): {location.values}, {round(float(observations_ds.lat[loc_idx].values))}, {round(float(observations_ds.lon[loc_idx].values))} - {round(float(observations_ds.altitude[loc_idx].values))} m.a.s) Time: {unixtime}, Observation: {round(float(observation.values)-273.15, 2)} ℃")

            # fetch forecast
            nearest_fcst = find_closest_value(ds=forecast_ds, lats=observations_ds.lat[loc_idx].values, lons=observations_ds.lon[loc_idx].values, unixtime=unixtime, variable="air_temperature_2m")
            if time_idx == 0:
                print("forecast", nearest_fcst-273.15, "℃", " alt adjusted", alt_correction(nearest_fcst-273.15, int(observations_ds.altitude[loc_idx].values)), "℃")

            locations.append(str(location.values))
            lats.append(float(observations_ds.lat[loc_idx].values))
            lons.append(float(observations_ds.lon[loc_idx].values))
            alts.append(int(observations_ds.altitude[loc_idx].values))
            time_list.append(int(unixtime))
            valid_dttm.append(int(unixtime)+3600) # fcst_dttm + lead_time
            obs.append(float(observation.values))
            fcst.append(alt_correction(float(nearest_fcst), int(observations_ds.altitude[loc_idx].values)))

    with tempfile.TemporaryDirectory() as temp_dir:
        pq.write_to_dataset(
            table = pa.table(
                data = [
                    time_list,
                    [3600 for x in range(len(time_list))],
                    valid_dttm,
                    locations,
                    lats,
                    lons,
                    alts,
                    [parameters[0] for x in range(len(time_list))],
                    ["K" for x in range(len(time_list))],
                    fcst,
                    obs,
                    [int(time.strftime('%H', time.gmtime(x))) for x in time_list],
                    [int(time.strftime('%Y', time.gmtime(x))) for x in time_list],
                    [int(time.strftime('%m', time.gmtime(x))) for x in time_list],
                    [int(time.strftime('%d', time.gmtime(x))) for x in time_list],
                    ],
                schema = pa.schema(
                    [
                    ("fcst_dttm", pa.int64()),
                    ("lead_time", pa.int32()),
                    ("valid_dttm", pa.int64()),
                    ("SID", pa.string()),
                    ("lat", pa.float64()),
                    ("lon", pa.float64()),
                    ("model_elevation", pa.int32()),
                    ("parameter", pa.string()),
                    ("units", pa.string()),
                    ("bris_det", pa.float64()),
                    ("obs", pa.float64()),
                    ("fcst_hour", pa.int32()),
                    ("fcst_year", pa.int32()),
                    ("fcst_month", pa.int32()),
                    ("fcst_day", pa.int32()),
                ]),
            ),
            root_path = temp_dir + outdir,
            existing_data_behavior="delete_matching",
            basename_template = "part-{i}.parquet",
            partitioning = pa_dataset.partitioning(
                pa.schema([
                    ("fcst_hour", pa.int32()),
                    ("fcst_year", pa.int32()),
                    ("fcst_month", pa.int32()),
                    ("fcst_day", pa.int32()),
                ]),
                flavor="hive",
            )
        )
        print(f"Wrote dataset to {temp_dir + outdir}. Now deleting it...")


if __name__ == "__main__":
    test_export_to_harp()
