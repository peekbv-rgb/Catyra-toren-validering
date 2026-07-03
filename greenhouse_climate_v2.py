#!/usr/bin/env python3
"""
Catyra Greenhouse Climate Model v2.0

Complete greenhouse climate simulation with:
- Tower physics from lookup tables or calculated
- VD-based tower control (hysteresis)
- VD-based roof vent control with outside RH protection
- Plant transpiration/humidification feedback
- Production damage scoring based on VD stress

Author: Catyra Engineering
Version: 2.0 (January 2026)

Usage:
    python greenhouse_climate_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TowerConfig:
    """Tower configuration for 3.2m scale tower."""
    H_tower: float = 3.2          # m - tower height
    D_top: float = 1.0            # m - top diameter (estimate)
    D_bottom: float = 1.0         # m - bottom diameter (estimate)
    H_discharge: float = 0.5      # m - discharge height above ground
    C_discharge: float = 0.61     # - discharge coefficient (validated 3.5m production tower)
    W_supply: float = 100.0       # L/hr - water supply to nozzles
    V_dot_evap_max: float = 25.0  # L/hr - MAX actual evaporation (physics-limited!)
    D_venturi: float = 0.95       # m - venturi/keeldiameter (referentie gevalideerde 3.5m toren)

    # Referentie-keel waarop de gevalideerde C=0.61 is geijkt (Technical Briefing dec 2025)
    D_VENTURI_REF: float = 0.95

    @property
    def A_bottom(self) -> float:
        """Outlet area [m²]."""
        return np.pi * (self.D_bottom / 2) ** 2

    @property
    def venturi_factor(self) -> float:
        """Venturi-correctie op het debiet, genormaliseerd op de gevalideerde keel (0.95 m).

        f = (D_venturi / D_VENTURI_REF)^2. Bij de gevalideerde keel is f = 1, zodat de
        gevalideerde prestatie exact reproduceerbaar blijft (geen dubbeltelling met C).
        """
        ref = self.D_VENTURI_REF if self.D_VENTURI_REF > 0 else 0.95
        return (self.D_venturi / ref) ** 2

    @property
    def A_eff(self) -> float:
        """Effectieve doorstroomoppervlak voor debiet/verdamping [m²] (venturi-gecorrigeerd)."""
        return self.A_bottom * self.venturi_factor


@dataclass
class GreenhouseConfig:
    """Greenhouse configuration."""
    A_floor: float = 10000.0      # m² - floor area (1 hectare)
    H_crop: float = 1.5           # m - crop zone height
    H_roof: float = 4.2           # m - roof height

    @property
    def V_crop_zone(self) -> float:
        """Crop zone volume [m³]."""
        return self.A_floor * self.H_crop

    @property
    def V_total(self) -> float:
        """Total greenhouse volume [m³]."""
        return self.A_floor * self.H_roof


@dataclass
class ControlConfig:
    """Control system configuration."""
    # Tower VD control (hysteresis) - OPTIMIZED thresholds
    # Testing showed ON>8 reduces damage from 64.7 to 40.5 pts (37% improvement)
    VD_tower_on: float = 8.0      # g/m³ - turn towers ON (was 9.0)
    VD_tower_off: float = 6.0     # g/m³ - turn towers OFF

    # Operating hours
    hour_start: int = 6           # Start hour (6 AM)
    hour_end: int = 20            # End hour (8 PM)

    # Roof VD control thresholds: {vd: roof_opening}
    roof_vd_thresholds: list = None

    # Outside RH protection limits: {rh: max_roof_opening}
    roof_rh_limits: list = None

    def __post_init__(self):
        # Default roof VD thresholds (VD -> roof opening fraction)
        if self.roof_vd_thresholds is None:
            self.roof_vd_thresholds = [
                {'vd': 10, 'roof': 0.00},  # Closed - protect humidity
                {'vd': 9, 'roof': 0.02},
                {'vd': 8, 'roof': 0.03},
                {'vd': 7, 'roof': 0.04},
                {'vd': 6, 'roof': 0.05},
                {'vd': 0, 'roof': 0.08},   # Well humidified - can ventilate
            ]

        # Default outside RH protection limits (RH -> max roof opening)
        if self.roof_rh_limits is None:
            self.roof_rh_limits = [
                {'rh': 40, 'max': 0.02},   # Very dry - minimal opening
                {'rh': 50, 'max': 0.04},
                {'rh': 60, 'max': 0.06},
                {'rh': 70, 'max': 0.08},
                {'rh': 100, 'max': 0.12},  # Humid outside - can open more
            ]


@dataclass
class CropConfig:
    """Crop configuration (strawberry)."""
    name: str = "strawberry"
    VD_optimal: float = 8.0       # g/m³ - 100% stomata open
    VD_stress: float = 10.0       # g/m³ - stress begins
    VD_critical: float = 12.0     # g/m³ - severe stress (~20% stomata)
    VD_lethal: float = 15.0       # g/m³ - production stops

    # Transpiration parameters
    ET_max: float = 0.5           # kg/m²/hr - max transpiration rate
    LAI: float = 3.0              # - leaf area index


@dataclass
class OrchardConfig:
    """Orchard configuration."""
    towers_per_ha: float = 25.0   # towers per hectare
    row_spacing: float = 3.5      # m - tree row spacing


@dataclass
class MixingConfig:
    """Mixing/stratification configuration."""
    stratification: float = 0.75  # - conservative stratification factor
    ACH_ref: float = 10.0         # /hr - reference ACH for saturation curve
    eta_mix: float = 0.23         # - mixing efficiency (climate room calibration)
    height_factor: float = 0.70   # - cooling effectiveness at crop height
    # GreenLight-validated separate T/RH mixing (set to None to use eta_mix for both)
    eta_temp: float = None        # - temperature mixing efficiency
    eta_rh: float = None          # - humidity mixing efficiency


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

G = 9.81            # m/s² - gravity
RHO_AIR = 1.2       # kg/m³ - air density (approximate)
CP_AIR = 1006       # J/(kg·K) - specific heat of air
L_VAP = 2.45e6      # J/kg - latent heat of vaporization
MW_WATER = 18.015   # g/mol - molecular weight of water
R_GAS = 8.314       # J/(mol·K) - gas constant


# =============================================================================
# PSYCHROMETRIC FUNCTIONS
# =============================================================================

def sat_vapor_pressure(T: float) -> float:
    """Saturation vapor pressure [Pa] using Magnus formula."""
    return 610.78 * np.exp(17.27 * T / (T + 237.3))


def vapor_deficit(T: float, RH: float) -> float:
    """Vapor deficit [g/m³]."""
    e_sat = sat_vapor_pressure(T)
    e_actual = e_sat * RH / 100
    return (e_sat - e_actual) * MW_WATER / (R_GAS * (T + 273.15))


def wet_bulb_temperature(T: float, RH: float) -> float:
    """Wet bulb temperature [°C] using Stull formula."""
    RH = np.clip(RH, 5, 99)
    T_wb = (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) - 4.686035)
    return T_wb


def humidity_ratio(T: float, RH: float) -> float:
    """Humidity ratio [kg water / kg dry air]."""
    e_sat = sat_vapor_pressure(T)
    e = e_sat * RH / 100
    P_atm = 101325  # Pa
    return 0.622 * e / (P_atm - e)


def RH_from_VD(T: float, VD: float) -> float:
    """Calculate RH from temperature and VD."""
    e_sat = sat_vapor_pressure(T)
    VD_sat = e_sat * MW_WATER / (R_GAS * (T + 273.15))
    e_actual = (VD_sat - VD) * R_GAS * (T + 273.15) / MW_WATER
    RH = 100 * e_actual / e_sat
    return np.clip(RH, 0, 100)


# =============================================================================
# TOWER PHYSICS
# =============================================================================

def calc_tower_velocity(T_amb: float, T_out: float, H: float, C: float) -> float:
    """
    Calculate tower outlet velocity [m/s].

    V = C × √(2 × g × H × ΔT / T_amb)
    """
    delta_T = T_amb - T_out
    if delta_T <= 0:
        return 0.0

    T_amb_K = T_amb + 273.15
    V = C * np.sqrt(2 * G * H * delta_T / T_amb_K)
    return V


def calc_tower_performance(T_amb: float, RH_amb: float,
                           tower: TowerConfig) -> Dict:
    """
    Calculate tower performance from physics.

    Returns dict with T_out, V_out, m_dot, V_dot, evap_rate
    """
    # Wet bulb temperature (tower outlet approaches this)
    T_wb = wet_bulb_temperature(T_amb, RH_amb)
    T_out = T_wb  # Evaporation-limited: outlet = wet bulb

    # Outlet velocity
    V_out = calc_tower_velocity(T_amb, T_out, tower.H_tower, tower.C_discharge)

    # Mass flow rate — debiet op de torendoorsnede (A_bottom).
    # Gevalideerd op 5 schaalmodel-reeksen (feb 2026): effectieve C = 0.66 ± 0.11 op de
    # torendoorsnede. De venturi versnelt de keelsnelheid maar bepaalt NIET het debiet,
    # daarom hier A_bottom en niet A_eff.
    m_dot = RHO_AIR * V_out * tower.A_bottom  # kg/s

    # Volume flow rate
    V_dot = m_dot / RHO_AIR * 3600  # m³/hr

    # Evaporation rate (physics-limited, NOT water supply limited!)
    w_amb = humidity_ratio(T_amb, RH_amb)
    w_sat = humidity_ratio(T_out, 100.0)
    delta_w = w_sat - w_amb

    # Evaporation = air mass flow × humidity pickup
    evap_rate = m_dot * delta_w * 3600  # kg/hr ≈ L/hr

    # Cap at physical maximum
    evap_rate = min(evap_rate, tower.V_dot_evap_max)

    return {
        'T_out': T_out,
        'T_wb': T_wb,
        'V_out': V_out,
        'm_dot': m_dot,
        'V_dot': V_dot,
        'evap_rate': evap_rate,
        'delta_T': T_amb - T_out,
    }


# =============================================================================
# CONTROL FUNCTIONS
# =============================================================================

def tower_vd_control(VD: float, current_state: bool,
                     control: ControlConfig) -> bool:
    """
    Tower VD control with hysteresis.

    ON when VD > VD_tower_on
    OFF when VD < VD_tower_off
    """
    if VD >= control.VD_tower_on:
        return True
    elif VD <= control.VD_tower_off:
        return False
    return current_state


def roof_vent_control(VD_inside: float, RH_outside: float,
                      control: ControlConfig = None) -> float:
    """
    VD-aware roof vent control with outside RH protection.

    Uses configurable thresholds from ControlConfig.

    Parameters
    ----------
    VD_inside : float
        Inside vapor deficit [g/m³]
    RH_outside : float
        Outside relative humidity [%]
    control : ControlConfig, optional
        Control configuration with thresholds

    Returns
    -------
    float
        Roof opening fraction [0-1]
    """
    if control is None:
        control = ControlConfig()

    # Interpolate roof opening from VD thresholds
    roof_base = _interpolate_threshold(
        VD_inside,
        control.roof_vd_thresholds,
        key_in='vd',
        key_out='roof',
        descending=True  # Higher VD = lower roof opening
    )

    # Interpolate max roof from outside RH
    roof_max = _interpolate_threshold(
        RH_outside,
        control.roof_rh_limits,
        key_in='rh',
        key_out='max',
        descending=False  # Higher RH = higher max opening
    )

    return min(roof_base, roof_max)


def _interpolate_threshold(value: float, thresholds: list,
                           key_in: str, key_out: str,
                           descending: bool = False) -> float:
    """
    Interpolate value from threshold list.

    Parameters
    ----------
    value : float
        Input value to look up
    thresholds : list
        List of {key_in: x, key_out: y} dicts
    key_in : str
        Key for input values
    key_out : str
        Key for output values
    descending : bool
        If True, thresholds are in descending order (high to low)

    Returns
    -------
    float
        Interpolated output value
    """
    # Sort thresholds
    sorted_thresh = sorted(thresholds, key=lambda x: x[key_in], reverse=descending)

    # Find bracketing values
    prev_in, prev_out = None, None

    for t in sorted_thresh:
        curr_in = t[key_in]
        curr_out = t[key_out]

        if descending:
            # For VD: 10, 9, 8, 7, 6, 0
            if value >= curr_in:
                return curr_out
        else:
            # For RH: 40, 50, 60, 70, 100
            if value <= curr_in:
                if prev_in is None:
                    return curr_out
                # Linear interpolation
                frac = (value - prev_in) / (curr_in - prev_in)
                return prev_out + frac * (curr_out - prev_out)

        prev_in, prev_out = curr_in, curr_out

    # Return last value if beyond range
    return sorted_thresh[-1][key_out]


# =============================================================================
# PLANT TRANSPIRATION
# =============================================================================

def calc_stomatal_response(VD: float, crop: CropConfig) -> float:
    """
    Stomatal opening factor [0-1] based on VD.

    f = 1 / (1 + ((VD - VD_optimal) / 7)²)  for VD > VD_optimal
    f = 1.0                                   for VD <= VD_optimal
    """
    if VD <= crop.VD_optimal:
        return 1.0
    return 1.0 / (1.0 + ((VD - crop.VD_optimal) / 7.0) ** 2)


def calc_plant_transpiration(T: float, RH: float, radiation: float,
                             f_stomata: float, crop: CropConfig,
                             gh: GreenhouseConfig) -> float:
    """
    Calculate plant transpiration rate [kg/hr].

    Based on Penman-Monteith simplified.
    """
    if radiation <= 50:  # Low light - minimal transpiration
        return 0.0

    # Simplified: ET proportional to radiation and stomatal opening
    # ET = ET_max × (R / 800) × f_stomata × LAI × A_floor
    ET_rate = crop.ET_max * (radiation / 800) * f_stomata * crop.LAI
    ET_total = ET_rate * gh.A_floor / 1000  # kg/hr for whole greenhouse

    return ET_total


# =============================================================================
# PRODUCTION DAMAGE SCORING
# =============================================================================

def calc_damage_score(VD: float, crop: CropConfig) -> float:
    """
    Calculate production damage score [0-100].

    0 = no damage (VD <= VD_optimal)
    100 = total loss (VD >= VD_lethal)
    """
    if VD <= crop.VD_optimal:
        return 0.0
    elif VD >= crop.VD_lethal:
        return 100.0
    elif VD <= crop.VD_stress:
        # 0-20% damage
        return 20 * (VD - crop.VD_optimal) / (crop.VD_stress - crop.VD_optimal)
    elif VD <= crop.VD_critical:
        # 20-60% damage
        return 20 + 40 * (VD - crop.VD_stress) / (crop.VD_critical - crop.VD_stress)
    else:
        # 60-100% damage
        return 60 + 40 * (VD - crop.VD_critical) / (crop.VD_lethal - crop.VD_critical)


def calc_daily_damage_points(hourly_damages: List[float]) -> float:
    """
    Calculate total daily damage points.

    Damage accumulates: 1 hour at 50% damage = 0.5 damage points
    """
    return sum(d / 100 for d in hourly_damages)


def get_damage_per_hour(vd: float) -> float:
    """Stepped damage scoring function - validated against CaTaVa data."""
    if vd <= 7: return 0
    elif vd <= 8: return 0.1
    elif vd <= 9: return 0.3
    elif vd <= 10: return 0.5
    elif vd <= 11: return 1.0
    elif vd <= 12: return 2.0
    elif vd <= 13: return 3.0
    elif vd <= 14: return 4.0
    elif vd <= 15: return 5.0
    elif vd <= 16: return 7.0
    elif vd <= 17: return 9.0
    elif vd <= 18: return 11.0
    elif vd <= 19: return 13.0
    elif vd <= 20: return 15.0
    elif vd <= 25: return 30.0
    else: return 45.0


# =============================================================================
# ORCHARD MODEL — 3-layer apple canopy with weighted VD
# =============================================================================

APPLE_LAYERS = {
    'L2': {'range': (0.5, 1.5), 'fraction': 0.481, 'f_tower': 0.60},
    'L3': {'range': (1.5, 2.5), 'fraction': 0.323, 'f_tower': 0.35},
    'L4': {'range': (2.5, 3.5), 'fraction': 0.196, 'f_tower': 0.15},
}


def orchard_model_hour(T_amb: float, RH_amb: float, radiation: float,
                       tower: TowerConfig, orchard: OrchardConfig,
                       crop: CropConfig, control: ControlConfig,
                       tower_state: bool) -> Dict:
    """Simulate one hour in an orchard with 3-layer canopy model."""
    VD_amb = vapor_deficit(T_amb, RH_amb)
    perf = calc_tower_performance(T_amb, RH_amb, tower)
    VD_tower = vapor_deficit(perf['T_out'], 100.0)

    tower_on = tower_vd_control(VD_amb, tower_state, control)

    VD_with = VD_amb
    T_crop = T_amb
    RH_with = RH_amb
    water = 0.0

    if tower_on:
        density = min(2.0, orchard.towers_per_ha / 25)
        VD_weighted = 0.0
        for layer in APPLE_LAYERS.values():
            f_adj = min(0.85, layer['f_tower'] * density)
            VD_layer = f_adj * VD_tower + (1 - f_adj) * VD_amb
            VD_weighted += VD_layer * layer['fraction']
        VD_with = VD_weighted
        T_crop = perf['T_out'] * 0.15 + T_amb * 0.85
        e_sat_crop = sat_vapor_pressure(T_crop)
        VD_sat = e_sat_crop * MW_WATER / (R_GAS * (T_crop + 273.15))
        RH_with = float(np.clip(100 * (1 - VD_with / VD_sat), 0, 100)) if VD_sat > 0 else RH_amb
        water = 43.2 * (orchard.towers_per_ha / 25)

    f_stomata = calc_stomatal_response(VD_with, crop)
    damage = get_damage_per_hour(VD_with)

    return {
        'T_baseline': T_amb, 'RH_baseline': RH_amb, 'VD_baseline': VD_amb,
        'T_crop': T_crop, 'RH_crop': RH_with, 'VD_crop': VD_with,
        'tower_on': tower_on, 'roof_pct': 0.0,
        'f_stomata': f_stomata, 'damage_baseline': get_damage_per_hour(VD_amb),
        'damage_towers': damage, 'water_L_hr': water,
        'cooling': T_amb - T_crop, 'evap_total': water,
        'radiation': radiation,
    }


def run_orchard_simulation(df: pd.DataFrame, tower: TowerConfig,
                           orchard: OrchardConfig, control: ControlConfig,
                           crop: CropConfig) -> pd.DataFrame:
    """Run full orchard simulation on a climate DataFrame."""
    results = []
    tower_state = False

    for _, row in df.iterrows():
        hour = row.get('hour', 12)
        if not (control.hour_start <= hour <= control.hour_end):
            tower_state = False

        res = orchard_model_hour(
            T_amb=row['T_inside'], RH_amb=row['RH_inside'],
            radiation=row.get('radiation', 400),
            tower=tower, orchard=orchard, crop=crop,
            control=control, tower_state=tower_state,
        )
        res['hour'] = hour
        res['datetime'] = row.get('datetime', None)
        res['date'] = row.get('date', None)
        tower_state = res['tower_on']
        results.append(res)

    return pd.DataFrame(results)


# =============================================================================
# GREENHOUSE MODEL
# =============================================================================

class GreenhouseClimateModel:
    """
    Complete greenhouse climate model with tower cooling.
    """

    def __init__(self,
                 n_towers: int = 32,
                 tower: TowerConfig = None,
                 greenhouse: GreenhouseConfig = None,
                 control: ControlConfig = None,
                 crop: CropConfig = None,
                 mixing: MixingConfig = None):

        self.n_towers = n_towers
        self.tower = tower or TowerConfig()
        self.greenhouse = greenhouse or GreenhouseConfig()
        self.control = control or ControlConfig()
        self.crop = crop or CropConfig()
        self.mixing = mixing or MixingConfig()

    def run_hourly(self, climate_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run hourly simulation.

        climate_data must have columns:
        - datetime or hour
        - T_inside (baseline temperature)
        - RH_inside (baseline humidity)
        - T_outside
        - RH_outside (or estimated)
        - radiation (optional)
        """
        results = []
        tower_state = False

        for idx, row in climate_data.iterrows():
            result = self._simulate_hour(row, tower_state)
            tower_state = result['tower_on']
            results.append(result)

        return pd.DataFrame(results)

    def _simulate_hour(self, row: pd.Series, tower_state: bool) -> Dict:
        """Simulate one hour."""

        # Extract conditions
        hour = row.get('hour', row.get('datetime', pd.Timestamp.now()).hour
                       if hasattr(row.get('datetime', 0), 'hour') else 12)
        T_baseline = row['T_inside']
        RH_baseline = row['RH_inside']
        T_outside = row.get('T_outside', T_baseline)
        RH_outside = row.get('RH_outside', 50)
        radiation = row.get('radiation', 400)

        # Baseline VD
        VD_baseline = vapor_deficit(T_baseline, RH_baseline)

        # Tower control
        tower_state = tower_vd_control(VD_baseline, tower_state, self.control)

        # Only operate during daytime
        if not (self.control.hour_start <= hour <= self.control.hour_end):
            tower_state = False

        # Initialize results
        T_crop = T_baseline
        RH_crop = RH_baseline
        VD_crop = VD_baseline
        cooling = 0.0
        evap_total = 0.0
        roof_pct = 0.0
        ACH = 0.0

        if tower_state:
            # Calculate tower performance
            perf = calc_tower_performance(T_outside, RH_outside, self.tower)

            # Total airflow from all towers
            V_dot_total = perf['V_dot'] * self.n_towers  # m³/hr
            ACH = V_dot_total / self.greenhouse.V_crop_zone

            # Mixing factor (saturates at high ACH)
            # This represents fraction of crop zone air that is "tower air"
            base_mix = self.mixing.stratification * (1 - np.exp(-ACH / self.mixing.ACH_ref))

            # Separate T and RH mixing factors (GreenLight vs single eta_mix)
            if self.mixing.eta_temp is not None and self.mixing.eta_rh is not None:
                # GreenLight mode: separate efficiency for T and RH
                f_mix_T = base_mix * (self.mixing.eta_temp / self.mixing.eta_mix) if self.mixing.eta_mix > 0 else base_mix
                f_mix_RH = base_mix * (self.mixing.eta_rh / self.mixing.eta_mix) if self.mixing.eta_mix > 0 else base_mix
            else:
                # Classic mode: single eta_mix for both
                f_mix_T = base_mix
                f_mix_RH = base_mix

            # Tower outlet conditions (saturated at wet bulb)
            T_tower = perf['T_out']
            w_tower = humidity_ratio(T_tower, 100.0)  # Saturated

            # Baseline conditions
            w_baseline = humidity_ratio(T_baseline, RH_baseline)

            # PROPER MIXING for temperature (uses f_mix_T)
            T_mixed = f_mix_T * T_tower + (1 - f_mix_T) * T_baseline

            # Apply height factor (cooling effectiveness at crop height)
            cooling = (T_baseline - T_mixed) * self.mixing.height_factor
            T_crop = T_baseline - cooling

            # PROPER MIXING for humidity (uses f_mix_RH)
            w_crop = f_mix_RH * w_tower + (1 - f_mix_RH) * w_baseline

            # Add evaporation contribution (small additional boost)
            evap_total = perf['evap_rate'] * self.n_towers  # L/hr
            # Additional humidity from evaporation not captured by mixing
            evap_boost = evap_total / self.greenhouse.V_crop_zone / RHO_AIR * 0.3
            w_crop = w_crop + evap_boost

            # Convert humidity ratio to RH at crop temperature
            w_sat_crop = humidity_ratio(T_crop, 100.0)
            RH_crop = min(85, 100 * w_crop / w_sat_crop)  # Max 85% RH in greenhouse

            # Calculate VD at crop
            VD_crop = vapor_deficit(T_crop, RH_crop)

        # Roof vent control
        roof_pct = roof_vent_control(VD_crop, RH_outside, self.control) * 100

        # Dry air intake effect (when roof open)
        if roof_pct > 1.0:
            VD_outside = vapor_deficit(T_outside, RH_outside)
            if VD_outside > VD_crop:
                # Dry air coming in - VD increases
                intake_factor = (roof_pct / 100) * 0.1
                VD_crop = VD_crop + intake_factor * (VD_outside - VD_crop)
                T_crop = T_crop + intake_factor * (T_outside - T_crop)
                RH_crop = RH_from_VD(T_crop, VD_crop)

        # Stomatal response
        f_stomata = calc_stomatal_response(VD_crop, self.crop)

        # Plant transpiration (adds humidity back)
        if radiation > 50:
            ET = calc_plant_transpiration(T_crop, RH_crop, radiation,
                                          f_stomata, self.crop, self.greenhouse)
            # Transpiration adds humidity
            if ET > 0:
                RH_boost_plant = ET / self.greenhouse.V_crop_zone * 1000 * 3
                RH_crop = min(85, RH_crop + RH_boost_plant)  # Max 85% RH
                VD_crop = vapor_deficit(T_crop, RH_crop)

        # Production damage
        damage_score = calc_damage_score(VD_crop, self.crop)

        return {
            'hour': hour,
            'T_outside': T_outside,
            'RH_outside': RH_outside,
            'T_baseline': T_baseline,
            'RH_baseline': RH_baseline,
            'VD_baseline': VD_baseline,
            'T_crop': T_crop,
            'RH_crop': RH_crop,
            'VD_crop': VD_crop,
            'cooling': cooling,
            'tower_on': tower_state,
            'n_towers': self.n_towers if tower_state else 0,
            'ACH': ACH,
            'evap_total': evap_total,
            'roof_pct': roof_pct,
            'f_stomata': f_stomata,
            'damage_score': damage_score,
            'radiation': radiation,
        }


# =============================================================================
# TOWER COMPARISON ANALYSIS
# =============================================================================

def run_tower_comparison(climate_data: pd.DataFrame,
                         tower_counts: List[int] = [26, 32, 38]) -> Dict:
    """
    Run comparison analysis for different tower counts.
    """
    results = {}

    for n_towers in tower_counts:
        print(f"\n  Running {n_towers} towers...")
        model = GreenhouseClimateModel(n_towers=n_towers)
        df = model.run_hourly(climate_data)

        # Calculate statistics
        day_hours = df[(df['hour'] >= 6) & (df['hour'] <= 20)]
        tower_on_hours = df[df['tower_on'] == True]

        stats = {
            'n_towers': n_towers,
            'df': df,

            # VD stats
            'VD_baseline_peak': df['VD_baseline'].max(),
            'VD_baseline_mean': day_hours['VD_baseline'].mean(),
            'VD_crop_peak': df['VD_crop'].max(),
            'VD_crop_mean': day_hours['VD_crop'].mean(),
            'VD_reduction_peak': df['VD_baseline'].max() - df['VD_crop'].max(),
            'VD_reduction_mean': day_hours['VD_baseline'].mean() - day_hours['VD_crop'].mean(),

            # Temperature stats
            'T_baseline_peak': df['T_baseline'].max(),
            'T_crop_peak': df['T_crop'].max(),
            'cooling_max': day_hours['cooling'].max(),
            'cooling_mean': day_hours[day_hours['cooling'] > 0]['cooling'].mean() if (day_hours['cooling'] > 0).any() else 0,

            # Operation stats
            'tower_hours': len(tower_on_hours),
            'tower_utilization': len(tower_on_hours) / len(day_hours) * 100 if len(day_hours) > 0 else 0,

            # Water usage
            'water_total': tower_on_hours['evap_total'].sum() / 1000,  # m³
            'water_per_hour': tower_on_hours['evap_total'].mean() if len(tower_on_hours) > 0 else 0,

            # Stomatal health
            'f_stomata_min': df['f_stomata'].min(),
            'f_stomata_mean': day_hours['f_stomata'].mean(),

            # Damage
            'damage_peak': df['damage_score'].max(),
            'damage_mean': day_hours['damage_score'].mean(),
            'damage_points': calc_daily_damage_points(df['damage_score'].tolist()),

            # Hours in stress zones
            'hours_VD_above_10': (df['VD_crop'] > 10).sum(),
            'hours_VD_above_12': (df['VD_crop'] > 12).sum(),
            'hours_optimal': ((df['VD_crop'] >= 4) & (df['VD_crop'] <= 8)).sum(),
        }

        results[n_towers] = stats

    return results


def print_comparison_report(results: Dict):
    """Print formatted comparison report."""

    print("\n" + "=" * 90)
    print("TOWER COMPARISON ANALYSIS")
    print("=" * 90)

    tower_counts = sorted(results.keys())

    # Header
    print(f"\n{'Metric':<35} |", end="")
    for n in tower_counts:
        print(f" {n:>10} towers |", end="")
    print()
    print("-" * 35 + "-+-" + "-+-".join(["-" * 13] * len(tower_counts)))

    # VD Metrics
    print(f"\n{'VD PERFORMANCE':<35}")
    metrics = [
        ('VD_baseline_peak', 'Baseline VD peak', 'g/m³'),
        ('VD_crop_peak', 'With towers VD peak', 'g/m³'),
        ('VD_reduction_peak', 'Peak VD reduction', 'g/m³'),
        ('VD_crop_mean', 'Mean VD (daytime)', 'g/m³'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<33} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>10.1f} {unit[:3]:>3}|", end="")
        print()

    # Temperature
    print(f"\n{'TEMPERATURE':<35}")
    metrics = [
        ('cooling_max', 'Max cooling', '°C'),
        ('cooling_mean', 'Mean cooling (when on)', '°C'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<33} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>10.1f} {unit[:3]:>3}|", end="")
        print()

    # Operation
    print(f"\n{'OPERATION':<35}")
    metrics = [
        ('tower_hours', 'Tower runtime', 'hrs'),
        ('tower_utilization', 'Utilization (daytime)', '%'),
        ('water_total', 'Water usage', 'm³'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<33} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>10.1f} {unit[:3]:>3}|", end="")
        print()

    # Plant Health
    print(f"\n{'PLANT HEALTH':<35}")
    metrics = [
        ('f_stomata_min', 'Min stomatal opening', '%×100'),
        ('f_stomata_mean', 'Mean stomatal opening', '%×100'),
        ('hours_optimal', 'Hours in optimal VD', 'hrs'),
        ('hours_VD_above_10', 'Hours VD > 10', 'hrs'),
        ('hours_VD_above_12', 'Hours VD > 12', 'hrs'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<33} |", end="")
        for n in tower_counts:
            val = results[n][key]
            if 'stomata' in key:
                val = val * 100  # Convert to percentage
            print(f" {val:>10.1f} {unit[:3]:>3}|", end="")
        print()

    # Damage
    print(f"\n{'PRODUCTION DAMAGE':<35}")
    metrics = [
        ('damage_peak', 'Peak damage score', '%'),
        ('damage_mean', 'Mean damage score', '%'),
        ('damage_points', 'Total damage points', 'pts'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<33} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>10.1f} {unit[:3]:>3}|", end="")
        print()

    print("\n" + "=" * 90)

    # Recommendation
    best_n = min(tower_counts, key=lambda n: results[n]['damage_points'])
    print(f"\nRECOMMENDATION: {best_n} towers")
    print(f"  - Lowest damage points: {results[best_n]['damage_points']:.2f}")
    print(f"  - Peak VD: {results[best_n]['VD_crop_peak']:.1f} g/m³")
    print(f"  - Water usage: {results[best_n]['water_total']:.1f} m³/day")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_greenhouse_data(filepath: str) -> pd.DataFrame:
    """Load greenhouse climate data from Excel."""

    df = pd.read_excel(filepath, header=None, skiprows=3)

    # Standard column names
    if len(df.columns) >= 8:
        df.columns = ['datetime', 'VD_measured', 'T_inside', 'RH_inside',
                      'T_outside', 'radiation', 'wind_speed', 'radiation_out'][:len(df.columns)]

    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour

    # Ensure numeric
    for col in df.columns:
        if col not in ['datetime']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Estimate outside RH if not available
    if 'RH_outside' not in df.columns:
        df['RH_outside'] = 50  # Default estimate

    return df


def create_hot_day_profile() -> pd.DataFrame:
    """Create synthetic hot day profile for testing."""

    hours = list(range(24))

    # Temperature profile (peaks at 14:00)
    T_base = 28
    T_amp = 10
    T_inside = [T_base + T_amp * np.sin(np.pi * (h - 6) / 16) if 6 <= h <= 22
                else T_base - 2 for h in hours]

    # RH profile (inverse of temperature)
    RH_base = 50
    RH_amp = 25
    RH_inside = [RH_base - RH_amp * np.sin(np.pi * (h - 6) / 16) if 6 <= h <= 22
                 else RH_base + 10 for h in hours]

    # Outside conditions
    T_outside = [t + 2 for t in T_inside]  # Slightly warmer outside
    RH_outside = [max(30, r - 10) for r in RH_inside]  # Drier outside

    # Radiation
    radiation = [0 if h < 6 or h > 20 else 800 * np.sin(np.pi * (h - 6) / 14)
                 for h in hours]

    return pd.DataFrame({
        'hour': hours,
        'T_inside': T_inside,
        'RH_inside': RH_inside,
        'T_outside': T_outside,
        'RH_outside': RH_outside,
        'radiation': radiation,
    })


def create_annual_climate_data() -> pd.DataFrame:
    """
    Create annual climate data for greenhouse in Mediterranean/Spanish climate.

    Based on typical Huelva, Spain strawberry growing region:
    - Growing season: October to June (main production)
    - Summer: Hot and dry (July-September)
    """

    data = []

    for day_of_year in range(1, 366):
        # Month estimation
        month = (day_of_year - 1) // 30 + 1
        month = min(12, max(1, month))

        # Seasonal temperature pattern (base temperature for each month)
        # Spanish greenhouse region - Huelva area
        monthly_T_base = {
            1: 12, 2: 14, 3: 17, 4: 20, 5: 24, 6: 28,
            7: 32, 8: 32, 9: 28, 10: 22, 11: 16, 12: 13
        }
        T_base = monthly_T_base[month]

        # Monthly RH pattern (base RH)
        monthly_RH_base = {
            1: 70, 2: 65, 3: 60, 4: 55, 5: 45, 6: 40,
            7: 35, 8: 35, 9: 40, 10: 55, 11: 65, 12: 70
        }
        RH_base = monthly_RH_base[month]

        # Daily variation amplitude (larger in summer)
        T_amp = 8 if month in [11, 12, 1, 2] else 12 if month in [6, 7, 8] else 10
        RH_amp = 15 if month in [11, 12, 1, 2] else 25 if month in [6, 7, 8] else 20

        # Radiation amplitude (higher in summer)
        rad_max = 400 if month in [11, 12, 1] else 600 if month in [2, 10] else 800 if month in [3, 4, 9] else 900

        for hour in range(24):
            # Diurnal patterns
            if 6 <= hour <= 22:
                phase = np.pi * (hour - 6) / 16
                T_inside = T_base + T_amp * np.sin(phase)
                RH_inside = max(25, RH_base - RH_amp * np.sin(phase))
            else:
                T_inside = T_base - 3
                RH_inside = min(85, RH_base + 10)

            # Outside slightly warmer/drier
            T_outside = T_inside + 2
            RH_outside = max(25, RH_inside - 10)

            # Radiation
            if 6 <= hour <= 20:
                radiation = rad_max * np.sin(np.pi * (hour - 6) / 14)
            else:
                radiation = 0

            data.append({
                'day_of_year': day_of_year,
                'month': month,
                'hour': hour,
                'T_inside': T_inside,
                'RH_inside': RH_inside,
                'T_outside': T_outside,
                'RH_outside': RH_outside,
                'radiation': radiation,
            })

    return pd.DataFrame(data)


def run_annual_simulation(n_towers: int = 32) -> Dict:
    """
    Run full annual simulation and calculate yearly statistics.
    """
    print(f"\n  Generating annual climate data...")
    climate_data = create_annual_climate_data()

    print(f"  Running simulation for {n_towers} towers over {len(climate_data)} hours...")
    model = GreenhouseClimateModel(n_towers=n_towers)
    results_df = model.run_hourly(climate_data)

    # Add day_of_year and month to results
    results_df['day_of_year'] = climate_data['day_of_year']
    results_df['month'] = climate_data['month']

    # Calculate annual statistics
    daytime = results_df[(results_df['hour'] >= 6) & (results_df['hour'] <= 20)]
    tower_on = results_df[results_df['tower_on'] == True]

    # Monthly breakdown
    monthly_stats = []
    for month in range(1, 13):
        month_data = results_df[results_df['month'] == month]
        month_day = month_data[(month_data['hour'] >= 6) & (month_data['hour'] <= 20)]
        month_tower = month_data[month_data['tower_on'] == True]

        monthly_stats.append({
            'month': month,
            'VD_baseline_mean': month_day['VD_baseline'].mean(),
            'VD_crop_mean': month_day['VD_crop'].mean(),
            'VD_reduction_mean': month_day['VD_baseline'].mean() - month_day['VD_crop'].mean(),
            'T_baseline_mean': month_day['T_baseline'].mean(),
            'cooling_mean': month_day['cooling'].mean(),
            'tower_hours': len(month_tower),
            'water_m3': month_tower['evap_total'].sum() / 1000,
            'damage_points': sum(d / 100 for d in month_data['damage_score']),
            'hours_VD_above_10': (month_data['VD_crop'] > 10).sum(),
            'hours_VD_above_12': (month_data['VD_crop'] > 12).sum(),
        })

    monthly_df = pd.DataFrame(monthly_stats)

    # Annual totals
    annual = {
        'n_towers': n_towers,

        # VD Performance
        'VD_baseline_peak': results_df['VD_baseline'].max(),
        'VD_baseline_mean': daytime['VD_baseline'].mean(),
        'VD_crop_peak': results_df['VD_crop'].max(),
        'VD_crop_mean': daytime['VD_crop'].mean(),
        'VD_reduction_mean': daytime['VD_baseline'].mean() - daytime['VD_crop'].mean(),
        'VD_reduction_pct': (1 - daytime['VD_crop'].mean() / daytime['VD_baseline'].mean()) * 100,

        # Operation
        'tower_hours_total': len(tower_on),
        'tower_hours_per_day': len(tower_on) / 365,
        'water_total_m3': tower_on['evap_total'].sum() / 1000,
        'water_per_day_m3': tower_on['evap_total'].sum() / 1000 / 365,

        # Plant Health
        'f_stomata_mean': daytime['f_stomata'].mean(),
        'hours_optimal_total': ((results_df['VD_crop'] >= 4) & (results_df['VD_crop'] <= 8)).sum(),
        'hours_VD_above_10': (results_df['VD_crop'] > 10).sum(),
        'hours_VD_above_12': (results_df['VD_crop'] > 12).sum(),

        # Damage
        'damage_points_total': sum(d / 100 for d in results_df['damage_score']),
        'damage_points_per_day': sum(d / 100 for d in results_df['damage_score']) / 365,

        # DataFrames
        'hourly_df': results_df,
        'monthly_df': monthly_df,
    }

    return annual


def run_annual_comparison(tower_counts: List[int] = [26, 32, 38]) -> Dict:
    """
    Run annual comparison for different tower counts.
    """
    results = {}

    for n_towers in tower_counts:
        print(f"\n--- {n_towers} Towers ---")
        results[n_towers] = run_annual_simulation(n_towers)

    return results


def print_annual_report(results: Dict):
    """Print annual performance report."""

    print("\n" + "=" * 100)
    print("ANNUAL PERFORMANCE COMPARISON")
    print("=" * 100)

    tower_counts = sorted(results.keys())

    # Header
    print(f"\n{'Metric':<45} |", end="")
    for n in tower_counts:
        print(f" {n:>12} towers |", end="")
    print()
    print("-" * 45 + "-+-" + "-+-".join(["-" * 15] * len(tower_counts)))

    # VD Performance
    print(f"\n{'VD PERFORMANCE (Annual)':<45}")
    metrics = [
        ('VD_baseline_mean', 'Baseline VD (daytime mean)', 'g/m³'),
        ('VD_crop_mean', 'With towers VD (daytime mean)', 'g/m³'),
        ('VD_reduction_mean', 'VD reduction (mean)', 'g/m³'),
        ('VD_reduction_pct', 'VD reduction (%)', '%'),
        ('VD_crop_peak', 'Peak VD (worst hour)', 'g/m³'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<43} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>12.1f} {unit[:4]:>4}|", end="")
        print()

    # Operation
    print(f"\n{'OPERATION (Annual)':<45}")
    metrics = [
        ('tower_hours_total', 'Total tower runtime', 'hrs'),
        ('tower_hours_per_day', 'Avg daily runtime', 'hrs'),
        ('water_total_m3', 'Total water usage', 'm³'),
        ('water_per_day_m3', 'Avg daily water', 'm³'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<43} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>12.1f} {unit[:4]:>4}|", end="")
        print()

    # Plant Health
    print(f"\n{'PLANT HEALTH (Annual)':<45}")
    metrics = [
        ('f_stomata_mean', 'Mean stomatal opening', '×100%'),
        ('hours_optimal_total', 'Hours in optimal VD (4-8)', 'hrs'),
        ('hours_VD_above_10', 'Hours VD > 10 (stress)', 'hrs'),
        ('hours_VD_above_12', 'Hours VD > 12 (severe)', 'hrs'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<43} |", end="")
        for n in tower_counts:
            val = results[n][key]
            if 'stomata' in key:
                val = val * 100
            print(f" {val:>12.0f} {unit[:4]:>4}|", end="")
        print()

    # Damage
    print(f"\n{'PRODUCTION DAMAGE (Annual)':<45}")
    metrics = [
        ('damage_points_total', 'Total damage points', 'pts'),
        ('damage_points_per_day', 'Avg daily damage', 'pts'),
    ]
    for key, label, unit in metrics:
        print(f"  {label:<43} |", end="")
        for n in tower_counts:
            print(f" {results[n][key]:>12.1f} {unit[:4]:>4}|", end="")
        print()

    print("\n" + "=" * 100)

    # Monthly breakdown for best option
    best_n = min(tower_counts, key=lambda n: results[n]['damage_points_total'])
    print(f"\nMONTHLY BREAKDOWN ({best_n} towers - recommended)")
    print("-" * 100)

    monthly = results[best_n]['monthly_df']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"{'Month':<8} | {'VD Base':>8} | {'VD Crop':>8} | {'Reduction':>10} | {'Tower hrs':>10} | {'Water m³':>10} | {'Damage pts':>11}")
    print("-" * 8 + "-+-" + "-+-".join(["-" * 10] * 6))

    for _, row in monthly.iterrows():
        print(f"{month_names[int(row['month'])-1]:<8} | {row['VD_baseline_mean']:>8.1f} | {row['VD_crop_mean']:>8.1f} | {row['VD_reduction_mean']:>10.1f} | {row['tower_hours']:>10.0f} | {row['water_m3']:>10.1f} | {row['damage_points']:>11.1f}")

    print("-" * 8 + "-+-" + "-+-".join(["-" * 10] * 6))
    print(f"{'TOTAL':<8} | {monthly['VD_baseline_mean'].mean():>8.1f} | {monthly['VD_crop_mean'].mean():>8.1f} | {monthly['VD_reduction_mean'].mean():>10.1f} | {monthly['tower_hours'].sum():>10.0f} | {monthly['water_m3'].sum():>10.1f} | {monthly['damage_points'].sum():>11.1f}")

    print("\n" + "=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("Catyra GREENHOUSE CLIMATE MODEL v2.0")
    print("=" * 90)

    # Try to load real data, fall back to synthetic
    data_path = Path('greenhousedata1.xlsx')

    if data_path.exists():
        print(f"\nLoading data from {data_path}...")
        df = load_greenhouse_data(data_path)

        # Filter to a hot day if possible
        if 'datetime' in df.columns:
            # Try July 2, 2025 (known hot day)
            hot_day = df[df['datetime'].dt.strftime('%Y-%m-%d') == '2025-07-02'].copy()
            if len(hot_day) > 0:
                print(f"  Using hot day: July 2, 2025 ({len(hot_day)} hours)")
                df = hot_day
            else:
                # Use first 24 hours
                df = df.head(24).copy()
                print(f"  Using first 24 hours of data")
    else:
        print("\nNo data file found, using synthetic hot day profile...")
        df = create_hot_day_profile()

    print(f"\nData summary:")
    print(f"  Hours: {len(df)}")
    print(f"  T_inside range: {df['T_inside'].min():.1f} - {df['T_inside'].max():.1f}°C")
    print(f"  RH_inside range: {df['RH_inside'].min():.1f} - {df['RH_inside'].max():.1f}%")

    # Run tower comparison
    print("\n" + "-" * 90)
    print("RUNNING TOWER COMPARISON: 26 / 32 / 38 towers")
    print("-" * 90)

    results = run_tower_comparison(df, tower_counts=[26, 32, 38])

    # Print report
    print_comparison_report(results)

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Save detailed results for each tower count
    for n_towers, stats in results.items():
        output_file = output_dir / f'greenhouse_results_{n_towers}_towers.csv'
        stats['df'].to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

    # Save summary
    summary_data = []
    for n_towers, stats in results.items():
        row = {k: v for k, v in stats.items() if k != 'df'}
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'tower_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    print("\n" + "=" * 90)
    print("MODEL RUN COMPLETE")
    print("=" * 90)

    return results


if __name__ == "__main__":
    results = main()
