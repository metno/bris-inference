"""Variables derived from model output (the `extra_variables` option of outputs)

A derived variable is computed from one or more anemoi variables at write time, so that it
can be requested from any output (netcdf, grib, verif, ...) without the model predicting it.
Supported names:

    ws          10 m wind speed from 10u and 10v
    wz_<level>  geometric vertical velocity [m/s] on the pressure level <level> [hPa], from
                omega (w_<level>, the Lagrangian pressure tendency [Pa/s] that IFS/ERA5
                datasets store as `w`), temperature (t_<level>) and specific humidity
                (q_<level>), using the hydrostatic hypothesis. Same formula and naming as the
                `w_to_wz` filter of anemoi-transform.
"""

from collections.abc import Callable

import numpy as np

R_DRY = 287.0  # gas constant of dry air [J/(kg K)], as in anemoi-transform
GRAVITY = 9.80665  # standard gravity [m/s^2]


def get_required_inputs(name: str) -> list[str] | None:
    """The anemoi variables needed to compute a derived variable

    Args:
        name: Variable name as used in a config (e.g. ws, wz_500)

    Returns:
        List of anemoi variable names, or None if `name` is not a derived variable
    """
    if name == "ws":
        return ["10u", "10v"]

    words = name.split("_")
    if len(words) == 2 and words[0] == "wz":
        level = _pressure_level(name)
        return [f"w_{level:d}", f"t_{level:d}", f"q_{level:d}"]

    return None


def is_derived(name: str) -> bool:
    return get_required_inputs(name) is not None


def compute(name: str, get: Callable[[str], np.ndarray]) -> np.ndarray:
    """Computes a derived variable

    Args:
        name: Derived variable name (see get_required_inputs)
        get: Function returning the array of a given anemoi variable. All arrays must have the
            same shape; the result has that shape too.

    Raises:
        ValueError: If there is no recipe for `name`
    """
    if name == "ws":
        return np.sqrt(get("10u") ** 2 + get("10v") ** 2)

    words = name.split("_")
    if len(words) == 2 and words[0] == "wz":
        level = _pressure_level(name)
        return omega_to_wz(
            get(f"w_{level:d}"), get(f"t_{level:d}"), get(f"q_{level:d}"), level
        )

    raise ValueError(f"No recipe to compute {name}")


def omega_to_wz(
    omega: np.ndarray,
    temperature: np.ndarray,
    humidity: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """Converts omega [Pa/s] to geometric vertical velocity wz [m/s]

    Hydrostatic hypothesis: dp/dz = -rho g, with the density of moist air from the ideal gas
    law using the virtual temperature T (1 + 0.61 q). Then wz = dz/dt = -omega / (rho g).
    Positive values are upward. This is the forward transform of anemoi-transform's `w_to_wz`
    filter (same constants and regularisation), so that datasets and inference agree.

    Args:
        omega: Lagrangian tendency of air pressure [Pa/s]
        temperature: Air temperature [K]
        humidity: Specific humidity [kg/kg]
        pressure_hpa: Pressure level [hPa]
    """
    rho = (100 * pressure_hpa) / (R_DRY * temperature * (1 + 0.61 * humidity) + 1e-8)
    return (-1.0 / (rho * GRAVITY + 1e-8)) * omega


def _pressure_level(name: str) -> int:
    level = name.split("_")[1]
    try:
        return int(level)
    except ValueError as e:
        raise ValueError(
            f"Invalid derived variable '{name}': the level must be an integer (e.g. wz_500)"
        ) from e
