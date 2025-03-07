import numpy as np
from earthkit.data.utils.dates import to_datetime
import datetime

"""
def phase_of_day(time) -> float:
    hour = time.astype(int)
    return hour * 2 * np.pi / 24

def phase_of_year(time) -> float:
    return (phase_of_day(time) - np.pi/2)/365.25

def cos_julian_day(time) -> float:
    return np.cos(phase_of_year(time))

def sin_julian_day(time) -> float:
    return np.sin(phase_of_year(time))

def local_phase_of_day(time, lons) -> np.ndarray:
    return phase_of_day(time) + lons * np.pi / 180.

def cos_local_time(time, lons) -> np.ndarray:
    return np.cos(local_phase_of_day(time, lons))

def sin_local_time(time, lons) -> np.ndarray:
    return np.sin(local_phase_of_day(time, lons))

def insolation(time, lats, lons) -> np.ndarray:
    jan_lat_shift_phase = 79*2*np.pi/365.25
    solar_latitude = 23.5 * np.pi/180 * np.sin(phase_of_year(time) - jan_lat_shift_phase)
    latitude_insolation = np.cos(lats * np.pi/180. - solar_latitude)
    longitude_insolation = -np.cos(local_phase_of_day(time, lons))
    latitude_insolation[latitude_insolation < 0] = 0
    longitude_insolation[longitude_insolation < 0] = 0

    return latitude_insolation * longitude_insolation  
"""


# This is copied from earthkit, probably need to declare this somewhere
def julian_day(date):
    date = to_datetime(date)
    delta = date - datetime.datetime(date.year, 1, 1)
    julian_day = delta.days + delta.seconds / 86400.0
    return julian_day


def cos_julian_day(date):
    radians = julian_day(date) / 365.25 * np.pi * 2
    return np.cos(radians)


def sin_julian_day(date):
    radians = julian_day(date) / 365.25 * np.pi * 2
    return np.sin(radians)


def local_time(date, lon):
    date = to_datetime(date)
    delta = date - datetime.datetime(date.year, date.month, date.day)
    hours_since_midnight = (delta.days + delta.seconds / 86400.0) * 24
    return (lon / 360.0 * 24.0 + hours_since_midnight) % 24


def cos_local_time(date, lon):
    radians = local_time(date, lon) / 24 * np.pi * 2
    return np.cos(radians)


def sin_local_time(date, lon):
    radians = local_time(date, lon) / 24 * np.pi * 2
    return np.sin(radians)


def insolation(date, lat, lon):
    return cos_solar_zenith_angle(date, lat, lon)


def toa_incident_solar_radiation(date, lat, lon):
    from earthkit.meteo.solar import toa_incident_solar_radiation

    date = to_datetime(date)
    result = toa_incident_solar_radiation(
        date - datetime.timedelta(minutes=30),
        date + datetime.timedelta(minutes=30),
        lat,
        lon,
        intervals_per_hour=2,
    )
    return result.flatten()


def cos_solar_zenith_angle(date, lat, lon):
    from earthkit.meteo.solar import cos_solar_zenith_angle

    date = to_datetime(date)
    result = cos_solar_zenith_angle(
        date,
        lat,
        lon,
    )
    return result.flatten()


def get_dynamic_forcings(time, lats, lons, selection):
    forcings = {}
    if "cos_julian_day" in selection:
        forcings["cos_julian_day"] = cos_julian_day(time)
    if "sin_julian_day" in selection:
        forcings["sin_julian_day"] = sin_julian_day(time)
    if "cos_local_time" in selection:
        forcings["cos_local_time"] = cos_local_time(time, lons)
    if "sin_local_time" in selection:
        forcings["sin_local_time"] = sin_local_time(time, lons)
    if "insolation" in selection:
        forcings["insolation"] = insolation(time, lats, lons)

    return forcings
