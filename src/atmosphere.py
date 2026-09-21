"""
AeroTwin-Nemotron: International Standard Atmosphere (ISA) Model
Physics-informed model calculating ambient temperature, pressure, and density.
"""

import math
from typing import Tuple, NamedTuple


class AtmosphereState(NamedTuple):
    ambient_temperature_c: float
    ambient_pressure_kpa: float
    air_density_kg_m3: float


def isa_atmosphere(
    altitude_m: float,
    temperature_offset_c: float = 0.0,
) -> AtmosphereState:
    """
    Computes ambient temperature, pressure, and air density based on the
    International Standard Atmosphere (ISA) model with altitude limits [0, 20000m].

    Parameters:
        altitude_m: Altitude above sea level in meters.
        temperature_offset_c: ISA temperature anomaly (e.g. ISA +10C).

    Returns:
        AtmosphereState with:
            ambient_temperature_c (deg C)
            ambient_pressure_kpa (kPa)
            air_density_kg_m3 (kg/m^3)
    """
    # Physical and aerodynamic constants
    T0 = 288.15        # Sea-level standard temperature (K)
    P0 = 101325.0      # Sea-level standard pressure (Pa)
    L = 0.0065         # Tropospheric lapse rate (K/m)
    R = 287.05287      # Specific gas constant for dry air (J/(kg*K))
    g = 9.80665        # Gravitational acceleration (m/s^2)
    H_TROPOPAUSE = 11000.0  # Tropopause altitude (m)

    # Clamping altitude physically
    h = max(0.0, min(20000.0, float(altitude_m)))

    if h <= H_TROPOPAUSE:
        # Troposphere standard temperature
        t_std = T0 - L * h
        p_std = P0 * (t_std / T0) ** (g / (R * L))
    else:
        # Stratosphere (isothermal up to 20,000m)
        t_trop = T0 - L * H_TROPOPAUSE  # 216.65 K
        p_trop = P0 * (t_trop / T0) ** (g / (R * L))
        t_std = t_trop
        p_std = p_trop * math.exp(-g * (h - H_TROPOPAUSE) / (R * t_trop))

    # Apply ambient temperature offset
    t_actual = t_std + float(temperature_offset_c)
    t_actual = max(180.0, t_actual)  # Prevent unphysical absolute zero

    # Air density using ideal gas law: rho = P / (R * T)
    density = p_std / (R * t_actual)

    ambient_temp_c = t_actual - 273.15
    ambient_pressure_kpa = p_std / 1000.0

    return AtmosphereState(
        ambient_temperature_c=round(ambient_temp_c, 2),
        ambient_pressure_kpa=round(ambient_pressure_kpa, 2),
        air_density_kg_m3=round(density, 4),
    )
