import numpy as np
import torch

def phase_of_day(time) -> np.float:
    hour = time.astype(int)
    return hour * 2 * np.pi / 24

def phase_of_year(time) -> np.float:
    return (phase_of_day(time) - np.pi/2)/365.25   

def cos_julian_day(time) -> np.float:
    return np.cos(phase_of_year(time))

def sin_julian_day(time) -> np.float:
    return np.sin(phase_of_year(time))

def local_phase_of_day(time, lons) -> np.ndarray:
    return phase_of_day(time) + lons

def cos_local_time(time, lons) -> np.ndarray:
    return np.cos(local_phase_of_day(time, lons))

def sin_local_time(time, lons) -> np.ndarray:
    return np.sin(local_phase_of_day(time, lons))

def insolation(time, lats, lons) -> np.ndarray:
    jan_lat_shift_phase = 79*2*np.pi/365
    solar_latitude = 23.5 * np.pi/180 * np.sin(phase_of_year(time) - jan_lat_shift_phase)
    latitude_insolation = np.cos(lats - solar_latitude)
    longitude_insolation = -np.cos(local_phase_of_day(time, lons))
    latitude_insolation[latitude_insolation < 0] = 0
    longitude_insolation[longitude_insolation < 0] = 0

    return latitude_insolation * longitude_insolation  

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




