#!/usr/bin/env python3
"""
Catyra - Interactive Web Dashboard

A comprehensive web tool for greenhouse climate simulation with evaporative cooling towers.
Uses the calibrated greenhouse_climate_v2.py model for accurate results.
Includes all production analysis visualizations.

Author: Catyra Engineering
Version: 2.0 (January 2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from io import BytesIO
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Try to import the actual model
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
    sys.path.insert(0, str(Path(__file__).parent))
    from greenhouse_climate_v2 import (
        GreenhouseClimateModel, TowerConfig, GreenhouseConfig, OrchardConfig,
        ControlConfig, CropConfig, MixingConfig, run_orchard_simulation,
        get_damage_per_hour as _get_damage_per_hour
    )
    USE_FULL_MODEL = True
except ImportError:
    USE_FULL_MODEL = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Catyra - Greenhouse Climate Simulator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3B82F6;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1E3A5F;
        border-left: 4px solid #3B82F6;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #3B82F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #EFF6FF;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS - Matching visualization scripts
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme matching reference images (from generate_visualizations.py)
COLORS = {
    'baseline': '#E74C3C',       # Red for without cooling
    'baseline_light': '#FADBD8', # Light red fill
    'cooling': '#3498DB',        # Blue for with cooling
    'cooling_light': '#D4E6F1',  # Light blue fill
    'optimal_zone': '#D5F5E3',   # Light green for optimal VD zone
    'stress_zone': '#FCF3CF',    # Light yellow for stress zone
    'danger_zone': '#FADBD8',    # Light red for danger zone
    'tower_on': '#2ECC71',       # Green for tower operation
    'roof_open': '#9B59B6',      # Purple for roof vents
    'production': '#8b5cf6',     # Purple for production
    # Aliases for backward compatibility
    'with_towers': '#3498DB',    # Same as cooling
    'towers_light': '#D4E6F1',   # Same as cooling_light
    'stress': '#f97316',         # Orange for stress thresholds
    'damage': '#dc2626',         # Red for damage thresholds
}


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
# ECONOMIC CONFIG
# =============================================================================

@dataclass
class EconomicConfig:
    damage_coeff: float = 0.083
    plants_per_m2: float = 6.0
    price_per_kg: float = 5.0
    greenhouse_area_ha: float = 1.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_climate_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and parse uploaded climate data file."""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=None, skiprows=3)
            cols = ['datetime', 'VD_measured', 'T_inside', 'RH_inside',
                    'T_outside', 'radiation', 'wind_speed', 'radiation_out']
            df.columns = cols[:len(df.columns)]
        else:
            # Try HTML tool forecast CSV format first (semicolon-separated, with header)
            df = _try_load_forecast_csv(uploaded_file)
            if df is not None:
                return df

            # Fall back to standard format
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None, skiprows=3)
            cols = ['datetime', 'VD_measured', 'T_inside', 'RH_inside',
                    'T_outside', 'radiation', 'wind_speed', 'radiation_out']
            df.columns = cols[:len(df.columns)]

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date

        for col in ['T_inside', 'RH_inside', 'T_outside', 'radiation', 'VD_measured']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'RH_outside' not in df.columns:
            df['RH_outside'] = (df['RH_inside'] - 10).clip(25, 90)

        df = df.dropna(subset=['datetime', 'T_inside', 'RH_inside'])
        return df
    except Exception as e:
        st.error(f"Error loading climate data: {e}")
        return None


def _try_load_forecast_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Try to load HTML tool forecast CSV (semicolon-separated with header)."""
    try:
        uploaded_file.seek(0)
        first_line = uploaded_file.readline()
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8')
        uploaded_file.seek(0)

        # Must be semicolon-separated
        if ';' not in first_line:
            return None

        df = pd.read_csv(uploaded_file, sep=';')

        # Map columns to standard names — support both Dutch and English formats
        col_map = {}
        for col in df.columns:
            cl = col.strip().lower()
            # Dutch format from HTML weather tool: Tijd;T_°C;RH_%;VD_gm3;...
            if cl in ('tijd',):
                col_map[col] = 'datetime'
            elif cl in ('t_°c', 't_c', 'temperature'):
                col_map[col] = 'T_inside'
            elif cl in ('rh_%', 'rh'):
                col_map[col] = 'RH_inside'
            elif cl in ('vd_gm3', 'vd'):
                col_map[col] = 'VD_measured'
            elif cl in ('straling_wm2', 'radiation', 'straling'):
                col_map[col] = 'radiation'
            elif cl in ('wind_ms', 'wind_speed', 'wind'):
                col_map[col] = 'wind_speed'
            elif cl in ('neerslag_mm', 'precipitation', 'neerslag'):
                col_map[col] = 'precipitation'
            # English format: datetime;T_inside;RH_inside;...
            elif cl == 'datetime':
                col_map[col] = 'datetime'
            elif cl == 't_inside':
                col_map[col] = 'T_inside'
            elif cl == 'rh_inside':
                col_map[col] = 'RH_inside'
            elif cl == 'vd_inside':
                col_map[col] = 'VD_measured'
            elif cl == 't_inside_lo':
                col_map[col] = 'T_inside_lo'
            elif cl == 't_inside_hi':
                col_map[col] = 'T_inside_hi'
            elif cl == 'vd_inside_lo':
                col_map[col] = 'VD_inside_lo'
            elif cl == 'vd_inside_hi':
                col_map[col] = 'VD_inside_hi'
            elif cl == 't_outside':
                col_map[col] = 'T_outside'
            elif cl == 'rh_outside':
                col_map[col] = 'RH_outside'

        df = df.rename(columns=col_map)

        if 'datetime' not in df.columns or 'T_inside' not in df.columns:
            return None

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date

        for col in ['T_inside', 'RH_inside', 'T_outside', 'radiation', 'VD_measured', 'RH_outside',
                    'T_inside_lo', 'T_inside_hi', 'VD_inside_lo', 'VD_inside_hi']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # For forecast data: outside = inside (it's outside weather data used as baseline)
        if 'T_outside' not in df.columns:
            df['T_outside'] = df['T_inside']
        if 'RH_outside' not in df.columns:
            df['RH_outside'] = df.get('RH_inside', pd.Series([50] * len(df)))

        if 'radiation' not in df.columns:
            df['radiation'] = 0

        df = df.dropna(subset=['datetime', 'T_inside', 'RH_inside'])
        return df
    except Exception:
        return None


def _try_load_historical_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Load multi-year historical CSV from HTML weather tool (semicolon-separated)."""
    try:
        uploaded_file.seek(0)
        first_line = uploaded_file.readline()
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8')
        uploaded_file.seek(0)

        if ';' not in first_line:
            return None

        df = pd.read_csv(uploaded_file, sep=';')

        col_map = {}
        for col in df.columns:
            cl = col.strip().lower()
            if cl == 'datetime': col_map[col] = 'datetime'
            elif cl == 't_inside': col_map[col] = 'T_inside'
            elif cl == 'rh_inside': col_map[col] = 'RH_inside'
            elif cl == 'vd_inside': col_map[col] = 'VD_measured'
            elif cl == 't_inside_lo': col_map[col] = 'T_inside_lo'
            elif cl == 't_inside_hi': col_map[col] = 'T_inside_hi'
            elif cl == 'vd_inside_lo': col_map[col] = 'VD_inside_lo'
            elif cl == 'vd_inside_hi': col_map[col] = 'VD_inside_hi'
            elif cl == 't_outside': col_map[col] = 'T_outside'
            elif cl == 'rh_outside': col_map[col] = 'RH_outside'
            elif cl in ('straling_wm2', 'radiation'): col_map[col] = 'radiation'
            elif cl in ('wind_ms', 'wind_speed'): col_map[col] = 'wind_speed'
            elif cl == 'year': col_map[col] = 'year'
            elif cl == 'month': col_map[col] = 'month'
            elif cl == 'hour': col_map[col] = 'hour'

        df = df.rename(columns=col_map)
        if 'T_inside' not in df.columns:
            return None

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if 'year' not in df.columns: df['year'] = df['datetime'].dt.year
            if 'month' not in df.columns: df['month'] = df['datetime'].dt.month
            if 'hour' not in df.columns: df['hour'] = df['datetime'].dt.hour
            df['date'] = df['datetime'].dt.date

        for col in ['T_inside', 'RH_inside', 'T_outside', 'radiation', 'VD_measured', 'RH_outside',
                    'T_inside_lo', 'T_inside_hi', 'VD_inside_lo', 'VD_inside_hi',
                    'year', 'month', 'hour']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'T_outside' not in df.columns: df['T_outside'] = df['T_inside']
        if 'RH_outside' not in df.columns:
            df['RH_outside'] = df.get('RH_inside', pd.Series([50] * len(df)))
        if 'radiation' not in df.columns: df['radiation'] = 0
        if 'year' not in df.columns: return None

        df = df.dropna(subset=['T_inside', 'RH_inside', 'year'])
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['hour'] = df['hour'].astype(int)
        return df
    except Exception:
        return None


def _hist_calc_vd(T, RH):
    """Calculate vapor deficit in g/m³."""
    e_sat = 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
    e_act = e_sat * RH / 100
    return (e_sat - e_act) * 1000 / (461.5 * (T + 273.15)) * 1000


def classify_historical_years(df: pd.DataFrame) -> Optional[Dict]:
    """Classify years as good/average/bad based on VD stress hours (growing season)."""
    grow = df[(df['month'].between(3, 9)) & (df['hour'].between(6, 19))].copy()

    if 'VD_measured' in grow.columns and grow['VD_measured'].notna().any():
        grow['vd'] = grow['VD_measured']
    else:
        grow['vd'] = _hist_calc_vd(grow['T_inside'], grow['RH_inside'])

    by_year = {}
    for year, grp in grow.groupby('year'):
        if len(grp) < 2000:
            continue
        stress10 = int((grp['vd'] > 10).sum())
        stress12 = int((grp['vd'] > 12).sum())
        by_year[int(year)] = {
            'year': int(year), 'stress10': stress10, 'stress12': stress12,
            'avgVD': float(grp['vd'].mean()), 'maxVD': float(grp['vd'].max()),
            'damage': float(grp['vd'].apply(get_damage_per_hour).sum()),
            'hours': len(grp),
        }

    if not by_year:
        return None

    ranked = sorted(by_year.values(), key=lambda x: x['stress10'])
    return {'good': ranked[0], 'avg': ranked[len(ranked)//2], 'bad': ranked[-1], 'ranked': ranked}


def _calc_tower_evap(T_amb: np.ndarray, RH_amb: np.ndarray,
                     H_tower: float = 3.5, C_discharge: float = 0.61,
                     D_bottom: float = 1.0) -> np.ndarray:
    """Calculate tower evaporation rate [L/hr] from ambient conditions.

    Physics: T_wb (Stull) → ΔT → V_out (buoyancy) → ṁ_air → Δw → evap.
    Returns array of evaporation rates in L/hr.
    """
    T = np.asarray(T_amb, dtype=float)
    RH = np.asarray(RH_amb, dtype=float)

    # Wet bulb temperature (Stull 2011 approximation)
    T_wb = (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) - 4.686035)

    # Tower outlet approaches wet bulb
    T_out = T_wb
    dT = np.maximum(T - T_out, 0.1)

    # Outlet velocity: V = C × sqrt(2 × g × H × ΔT / T_amb_K)
    g = 9.81
    T_K = T + 273.15
    V_out = C_discharge * np.sqrt(2 * g * H_tower * dT / T_K)

    # Mass flow through tower bottom
    A_bottom = np.pi * (D_bottom / 2) ** 2
    rho_air = 1.2  # kg/m³ (approximate)
    m_dot = rho_air * V_out * A_bottom * 3600  # kg/hr

    # Humidity ratio at ambient and at saturation (tower outlet)
    P_atm = 101325  # Pa
    # Ambient: e_actual → w_amb
    e_sat_amb = 610.78 * np.exp(17.27 * T / (T + 237.3))
    e_actual = e_sat_amb * RH / 100
    w_amb = 0.622 * e_actual / (P_atm - e_actual)

    # Outlet (saturated at T_out): e_sat → w_sat
    e_sat_out = 610.78 * np.exp(17.27 * T_out / (T_out + 237.3))
    w_sat = 0.622 * e_sat_out / (P_atm - e_sat_out)

    # Evaporation = mass_flow × (w_sat - w_amb)
    delta_w = np.maximum(w_sat - w_amb, 0)
    evap = m_dot * delta_w  # L/hr (water density ≈ 1 kg/L)

    return np.clip(evap, 0, 50)  # physical cap


def calc_tower_perf(T_amb, RH_amb, H_tower: float = 3.5, C_discharge: float = 0.61,
                    D_bottom: float = 1.0, evap_max: float = 25.0) -> Dict:
    """Volledige performance per toren vanuit omgevingscondities.

    Fysica identiek aan greenhouse_climate_v2.calc_tower_performance:
    T_wb (Stull) -> T_out -> ΔT -> V_out (schoorsteeneffect) -> ṁ -> V̇ (luchthoeveelheid) -> verdamping.
    Werkt op scalars en numpy-arrays. Retourneert een dict met numpy-waarden.
    """
    T = np.asarray(T_amb, dtype=float)
    RH = np.asarray(RH_amb, dtype=float)

    # Natteboltemperatuur (Stull 2011); uitlaat koelt af tot ~natbol
    T_wb = (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) - 4.686035)
    T_out = T_wb
    dT = np.maximum(T - T_out, 0.0)

    # Uitlaatsnelheid (buoyancy / schoorsteeneffect)
    g = 9.81
    T_K = T + 273.15
    V_out = C_discharge * np.sqrt(2 * g * H_tower * dT / T_K)      # m/s

    # Luchthoeveelheid
    A_outlet = np.pi * (D_bottom / 2) ** 2                        # m²
    rho_air = 1.2                                                 # kg/m³
    m_dot = rho_air * V_out * A_outlet                           # kg/s
    V_dot_s = V_out * A_outlet                                   # m³/s
    V_dot_hr = V_dot_s * 3600.0                                  # m³/hr per toren

    # Verdamping (fysisch begrensd, niet watertoevoer-begrensd)
    P_atm = 101325.0
    e_sat_amb = 610.78 * np.exp(17.27 * T / (T + 237.3))
    e_actual = e_sat_amb * RH / 100
    w_amb = 0.622 * e_actual / (P_atm - e_actual)
    e_sat_out = 610.78 * np.exp(17.27 * T_out / (T_out + 237.3))
    w_sat = 0.622 * e_sat_out / (P_atm - e_sat_out)
    delta_w = np.maximum(w_sat - w_amb, 0.0)
    evap = np.minimum(m_dot * delta_w * 3600.0, evap_max)        # L/hr

    # Werkelijke vochtopname op basis van (begrensde) verdamping -> uitblaascondities
    with np.errstate(divide='ignore', invalid='ignore'):
        w_pickup = np.where(m_dot > 0, evap / (m_dot * 3600.0), 0.0)  # kg/kg
    w_out = w_amb + w_pickup
    e_out = w_out * P_atm / (0.622 + w_out)
    RH_out = np.clip(100.0 * e_out / e_sat_out, 0.0, 100.0)      # % (verzadigd -> ~100)

    # Koelvermogen uit verdamping
    L_vap = 2.45e6                                               # J/kg
    cooling_kW = evap / 3600.0 * L_vap / 1000.0                  # kW

    return {
        'T_wb': T_wb, 'T_out': T_out, 'dT': dT,
        'V_out': V_out, 'A_outlet': A_outlet, 'm_dot': m_dot,
        'V_dot_s': V_dot_s, 'V_dot_hr': V_dot_hr,
        'RH_out': RH_out, 'w_amb': w_amb, 'w_out': w_out,
        'evap': evap, 'cooling_kW': cooling_kW,
    }


def compute_historical_vd_stats(df: pd.DataFrame,
                                H_tower: float = 3.5, C_discharge: float = 0.61,
                                D_bottom: float = 1.0) -> Dict:
    """Compute daily VD statistics across 10 years for percentile analysis.

    Returns dict with:
      - daily_percentiles: DataFrame with day_of_year, P50, P75, P90, P95, P99 of daily max VD
      - avg_year_profile: daily max VD averaged across all years
      - worst_year: dict with year label and its daily max VD profile
      - best_year: dict with year label and its daily max VD profile
      - year_summaries: per-year summary (total stress, max VD, etc.)
    """
    hdf = df.copy()

    # Calculate VD if not present
    if 'VD_measured' in hdf.columns and hdf['VD_measured'].notna().any():
        hdf['vd'] = hdf['VD_measured']
    elif 'VD_inside' in hdf.columns and hdf['VD_inside'].notna().any():
        hdf['vd'] = pd.to_numeric(hdf['VD_inside'], errors='coerce')
    else:
        hdf['vd'] = _hist_calc_vd(hdf['T_inside'], hdf['RH_inside'])

    # Ensure datetime and day_of_year
    if 'datetime' in hdf.columns:
        hdf['datetime'] = pd.to_datetime(hdf['datetime'], errors='coerce')
        hdf['day_of_year'] = hdf['datetime'].dt.dayofyear
    else:
        # Reconstruct from year/month/hour
        hdf['day_of_year'] = pd.to_datetime(
            hdf['year'].astype(int).astype(str) + '-' + hdf['month'].astype(int).astype(str) + '-01',
            errors='coerce'
        ).dt.dayofyear + (hdf['hour'] / 24)
        hdf['day_of_year'] = hdf['day_of_year'].astype(int)

    if 'date' not in hdf.columns:
        hdf['date'] = hdf['datetime'].dt.date if 'datetime' in hdf.columns else None

    # Growing season only: daytime hours (6-20h), months 3-9
    grow = hdf[(hdf['month'].between(3, 9)) & (hdf['hour'].between(6, 20))].copy()

    # Calculate tower evaporation for each hour (based on OUTSIDE conditions)
    T_out_col = 'T_outside' if 'T_outside' in grow.columns else 'T_inside'
    RH_out_col = 'RH_outside' if 'RH_outside' in grow.columns else 'RH_inside'
    grow['evap_rate'] = _calc_tower_evap(
        grow[T_out_col].values, grow[RH_out_col].values,
        H_tower=H_tower, C_discharge=C_discharge, D_bottom=D_bottom
    )

    # Daily max VD and max evaporation per year per day_of_year
    daily_max = grow.groupby(['year', 'day_of_year']).agg(
        vd_max=('vd', 'max'),
        evap_max=('evap_rate', 'max'),
    ).reset_index()

    # VD percentiles across years for each day_of_year
    pcts = daily_max.groupby('day_of_year')['vd_max'].agg(
        P50='median',
        P75=lambda x: np.percentile(x, 75),
        P90=lambda x: np.percentile(x, 90),
        P95=lambda x: np.percentile(x, 95),
        P99=lambda x: np.percentile(x, 99),
        mean='mean',
    ).reset_index()

    # Evaporation percentiles across years for each day_of_year
    evap_pcts = daily_max.groupby('day_of_year')['evap_max'].agg(
        P50='median',
        P75=lambda x: np.percentile(x, 75),
        P90=lambda x: np.percentile(x, 90),
        P95=lambda x: np.percentile(x, 95),
        P99=lambda x: np.percentile(x, 99),
        mean='mean',
    ).reset_index()

    # Average year profile (mean of daily max across years)
    avg_profile = daily_max.groupby('day_of_year')['vd_max'].mean().reset_index()
    avg_profile.columns = ['day_of_year', 'vd_max_avg']

    # Year summaries to find worst/best
    year_stats = {}
    for yr, grp in daily_max.groupby('year'):
        year_stats[int(yr)] = {
            'year': int(yr),
            'total_vd_max': float(grp['vd_max'].sum()),
            'peak_vd': float(grp['vd_max'].max()),
            'peak_evap': float(grp['evap_max'].max()),
            'stress_days_10': int((grp['vd_max'] > 10).sum()),
            'stress_days_12': int((grp['vd_max'] > 12).sum()),
            'avg_daily_max': float(grp['vd_max'].mean()),
            'profile': grp[['day_of_year', 'vd_max']].copy(),
        }

    ranked = sorted(year_stats.values(), key=lambda x: x['total_vd_max'])
    worst = ranked[-1]
    best = ranked[0]
    median_yr = ranked[len(ranked) // 2]

    return {
        'daily_percentiles': pcts,
        'evap_percentiles': evap_pcts,
        'avg_profile': avg_profile,
        'worst_year': worst,
        'best_year': best,
        'median_year': median_yr,
        'year_stats': year_stats,
        'ranked': ranked,
    }


def create_historical_vd_chart(stats: Dict, nozzle_flow: float = 1.6) -> plt.Figure:
    """Create 4-panel chart: VD percentiles, avg/worst year, evap rate + nozzles, year ranking."""
    pcts = stats['daily_percentiles']
    evap_pcts = stats['evap_percentiles']
    avg_prof = stats['avg_profile']
    worst = stats['worst_year']
    best = stats['best_year']
    median_yr = stats['median_year']

    month_starts = [60, 91, 121, 152, 182, 213, 244, 274]
    month_names = ['Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt']

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [1.2, 1, 1.2, 0.7]})

    # --- Panel 1: VD Percentile bands ---
    ax = axes[0]
    x = pcts['day_of_year']
    ax.fill_between(x, 0, pcts['P50'], alpha=0.15, color='#22c55e', label='P50 (mediaan)')
    ax.fill_between(x, pcts['P50'], pcts['P75'], alpha=0.2, color='#eab308', label='P50–P75')
    ax.fill_between(x, pcts['P75'], pcts['P90'], alpha=0.25, color='#f97316', label='P75–P90')
    ax.fill_between(x, pcts['P90'], pcts['P95'], alpha=0.3, color='#ef4444', label='P90–P95')
    ax.fill_between(x, pcts['P95'], pcts['P99'], alpha=0.35, color='#dc2626', label='P95–P99')
    ax.plot(x, pcts['P95'], color='#dc2626', linewidth=1.5, linestyle='--', label='P95')
    ax.plot(x, pcts['P99'], color='#7f1d1d', linewidth=1.5, linestyle=':', label='P99')
    ax.axhline(y=10, color='orange', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axhline(y=12, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.text(60, 10.2, 'stress drempel', fontsize=8, color='orange')
    ax.text(60, 12.2, 'kritiek', fontsize=8, color='red')
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    ax.set_xlim(60, 274)
    ax.set_ylabel('Max VD per dag (g/m³)')
    ax.set_title('Dagelijks max VD — Percentielen over 10 jaar', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(pcts['P99'].max() * 1.1, 16))

    # --- Panel 2: Average vs worst vs median year ---
    ax2 = axes[1]
    ax2.plot(avg_prof['day_of_year'], avg_prof['vd_max_avg'], color='#3b82f6', linewidth=2,
             label=f'Gemiddeld jaar (n={len(stats["year_stats"])})')
    wp = worst['profile']
    ax2.plot(wp['day_of_year'], wp['vd_max'], color='#ef4444', linewidth=1.5, alpha=0.8,
             label=f'Slechtste jaar ({worst["year"]})')
    mp = median_yr['profile']
    ax2.plot(mp['day_of_year'], mp['vd_max'], color='#22c55e', linewidth=1.5, alpha=0.8,
             label=f'Mediaan jaar ({median_yr["year"]})')
    ax2.axhline(y=10, color='orange', linewidth=0.8, linestyle=':', alpha=0.7)
    ax2.axhline(y=12, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels(month_names)
    ax2.set_xlim(60, 274)
    ax2.set_ylabel('Max VD per dag (g/m³)')
    ax2.set_title('Gemiddeld, mediaan en slechtste jaar — dagelijks max VD', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Tower evaporation rate percentiles + nozzle count ---
    ax3 = axes[2]
    xe = evap_pcts['day_of_year']
    ax3.fill_between(xe, 0, evap_pcts['P50'], alpha=0.15, color='#0ea5e9', label='P50')
    ax3.fill_between(xe, evap_pcts['P50'], evap_pcts['P75'], alpha=0.2, color='#0284c7')
    ax3.fill_between(xe, evap_pcts['P75'], evap_pcts['P90'], alpha=0.25, color='#0369a1')
    ax3.fill_between(xe, evap_pcts['P90'], evap_pcts['P95'], alpha=0.3, color='#1e40af', label='P90–P95')
    ax3.fill_between(xe, evap_pcts['P95'], evap_pcts['P99'], alpha=0.35, color='#1e3a8a', label='P95–P99')
    ax3.plot(xe, evap_pcts['P95'], color='#1e3a8a', linewidth=2, linestyle='--', label='P95 (ontwerp)')
    ax3.plot(xe, evap_pcts['P99'], color='#312e81', linewidth=1.5, linestyle=':', label='P99')
    ax3.plot(xe, evap_pcts['mean'], color='#0ea5e9', linewidth=2, label='Gemiddeld')

    ax3.set_xticks(month_starts)
    ax3.set_xticklabels(month_names)
    ax3.set_xlim(60, 274)
    ax3.set_ylabel('Verdamping per toren (L/uur)')
    ax3.set_title('Vereiste verdampingscapaciteit per toren — Percentielen', fontweight='bold', fontsize=13)
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Secondary y-axis: nozzle count (Foggy Home @ nozzle_flow L/hr)
    ax3b = ax3.twinx()
    evap_max = max(evap_pcts['P99'].max(), 1)
    nozzle_max = int(np.ceil(evap_max / nozzle_flow)) + 2
    ax3b.set_ylim(0, nozzle_max * nozzle_flow)
    nozzle_ticks = list(range(0, nozzle_max + 1, max(1, nozzle_max // 8)))
    ax3b.set_yticks([n * nozzle_flow for n in nozzle_ticks])
    ax3b.set_yticklabels([str(n) for n in nozzle_ticks])
    ax3b.set_ylabel(f'Aantal nozzles ({nozzle_flow} L/uur per stuk)')

    # Highlight design point: P95 peak
    p95_peak_evap = evap_pcts['P95'].max()
    p95_peak_doy = evap_pcts.loc[evap_pcts['P95'].idxmax(), 'day_of_year']
    n_nozzles_p95 = int(np.ceil(p95_peak_evap / nozzle_flow))
    ax3.annotate(f'{p95_peak_evap:.1f} L/hr\n= {n_nozzles_p95} nozzles',
                 xy=(p95_peak_doy, p95_peak_evap), fontsize=9, fontweight='bold',
                 xytext=(p95_peak_doy + 15, p95_peak_evap + 1),
                 arrowprops=dict(arrowstyle='->', color='#1e3a8a'),
                 color='#1e3a8a', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # --- Panel 4: Year ranking bar chart ---
    ax4 = axes[3]
    ranked = stats['ranked']
    years = [r['year'] for r in ranked]
    totals = [r['avg_daily_max'] for r in ranked]
    colors = ['#ef4444' if r['year'] == worst['year']
              else '#22c55e' if r['year'] == best['year']
              else '#3b82f6' for r in ranked]
    ax4.bar(range(len(years)), totals, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(years)))
    ax4.set_xticklabels(years, rotation=45)
    ax4.set_ylabel('Gem. dagelijks max VD (g/m³)')
    ax4.set_title('Jaarrangschikking — gemiddeld dagelijks maximum VD', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(pad=2)
    return fig


def load_production_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load production data file."""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=None)
        else:
            df = pd.read_csv(uploaded_file, header=None)

        production_data = []
        for i in range(7, min(100, len(df))):
            try:
                date = df.iloc[i, 0]
                cumulative = df.iloc[i, 15] if len(df.columns) > 15 else df.iloc[i, -1]
                if pd.notna(date) and pd.notna(cumulative):
                    production_data.append({
                        'date': pd.to_datetime(date),
                        'cumulative_g': float(cumulative)
                    })
            except:
                continue

        if production_data:
            prod_df = pd.DataFrame(production_data)
            prod_df['cumulative_kg'] = prod_df['cumulative_g'] / 1000
            return prod_df
        return None
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return None


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(df: pd.DataFrame, n_towers: int, tower, greenhouse, control, crop, mixing,
                   is_greenhouse: bool = True, orchard=None) -> pd.DataFrame:
    """Run simulation using the calibrated model."""
    if not is_greenhouse and USE_FULL_MODEL and orchard is not None:
        # Orchard model
        results = run_orchard_simulation(df, tower, orchard, control, crop)
    elif USE_FULL_MODEL:
        # Greenhouse model
        model = GreenhouseClimateModel(
            n_towers=n_towers, tower=tower, greenhouse=greenhouse,
            control=control, crop=crop, mixing=mixing
        )
        results = model.run_hourly(df)
        results['datetime'] = df['datetime'].values
        results['date'] = df['date'].values
        results['hour'] = df['hour'].values
    else:
        results = run_simplified_simulation(df, n_towers, tower, control, greenhouse, mixing)

    # Add damage calculations if not already present
    if 'damage_baseline' not in results.columns:
        results['damage_baseline'] = results['VD_baseline'].apply(get_damage_per_hour)
    if 'damage_towers' not in results.columns:
        results['damage_towers'] = results['VD_crop'].apply(get_damage_per_hour)

    # Carry through RF confidence bands if available
    for col in ['T_inside_lo', 'T_inside_hi', 'VD_inside_lo', 'VD_inside_hi']:
        if col in df.columns:
            results[col] = df[col].values

    return results


def run_simplified_simulation(df, n_towers, tower, control, greenhouse, mixing):
    """Fallback simplified simulation."""
    results = []
    tower_was_on = False
    g = 9.81

    for _, row in df.iterrows():
        T_in = row['T_inside']
        RH_in = row['RH_inside']
        hour = row['hour']

        e_sat = 0.61121 * np.exp((18.678 - T_in / 234.5) * (T_in / (257.14 + T_in)))
        e_actual = e_sat * RH_in / 100
        VD_baseline = (e_sat - e_actual) * 1000 / (461.5 * (T_in + 273.15)) * 1000

        T_wb = T_in * np.arctan(0.151977 * np.sqrt(RH_in + 8.313659)) + \
               np.arctan(T_in + RH_in) - np.arctan(RH_in - 1.676331) + \
               0.00391838 * (RH_in ** 1.5) * np.arctan(0.023101 * RH_in) - 4.686035

        in_hours = control.hour_start <= hour <= control.hour_end
        if tower_was_on:
            tower_on = VD_baseline > control.VD_tower_off and in_hours
        else:
            tower_on = VD_baseline > control.VD_tower_on and in_hours

        T_crop, RH_crop, VD_crop = T_in, RH_in, VD_baseline
        cooling, evap_total = 0.0, 0.0

        if tower_on and n_towers > 0:
            delta_T = T_in - T_wb
            if delta_T > 0.5:
                V = tower.C_discharge * np.sqrt(2 * g * tower.H_tower * delta_T / (T_in + 273.15))
                V = min(V, 3.0)
                A = np.pi * (tower.D_top / 2) ** 2
                m_dot = 1.2 * V * A

                w_in = 0.622 * (e_sat * RH_in / 100) / (101.325 - e_sat * RH_in / 100)
                e_sat_wb = 0.61121 * np.exp((18.678 - T_wb / 234.5) * (T_wb / (257.14 + T_wb)))
                w_sat = 0.622 * e_sat_wb / (101.325 - e_sat_wb)
                delta_w = max(0, (w_sat - w_in) * 0.85)

                evap_L_hr = min(m_dot * delta_w * 3600, tower.V_dot_evap_max)
                evap_total = evap_L_hr * n_towers

                cooling_kW = evap_L_hr * n_towers * 2450 / 3600 / 1000
                gh_volume = getattr(greenhouse, 'volume', greenhouse.A_floor * getattr(greenhouse, 'H_roof', 4.2))
                m_air = gh_volume * 1.2
                eta = mixing.eta_mix * mixing.stratification
                cooling = cooling_kW * 3600 / (m_air * 1.005) * eta
                cooling = min(cooling, delta_T * 0.8) * mixing.height_factor

                T_crop = T_in - cooling
                moisture = evap_total / 1000
                rh_max = getattr(greenhouse, 'RH_max', 85)
                RH_crop = min(rh_max, RH_in + moisture / (m_air * 0.001) * 100 * 0.3)

                e_sat_crop = 0.61121 * np.exp((18.678 - T_crop / 234.5) * (T_crop / (257.14 + T_crop)))
                e_actual_crop = e_sat_crop * RH_crop / 100
                VD_crop = (e_sat_crop - e_actual_crop) * 1000 / (461.5 * (T_crop + 273.15)) * 1000

        results.append({
            'datetime': row['datetime'], 'date': row['date'], 'hour': hour,
            'T_baseline': T_in, 'T_crop': T_crop,
            'RH_baseline': RH_in, 'RH_crop': RH_crop,
            'VD_baseline': VD_baseline, 'VD_crop': VD_crop,
            'cooling': cooling, 'evap_total': evap_total, 'tower_on': tower_on,
        })
        tower_was_on = tower_on

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS - Exact copies from generate_visualizations.py
# =============================================================================

def create_hot_day_4panel(results_df: pd.DataFrame, date, n_towers: int, tower_height: float) -> plt.Figure:
    """
    Create 4-panel hot day analysis plot - EXACT MATCH to generate_visualizations.py

    Panels:
    1. Temperature Control
    2. VD Control
    3. Humidity Control
    4. Control Timeline
    """
    day_data = results_df[results_df['date'] == date].copy()
    if len(day_data) == 0:
        return None

    # CRITICAL: Sort by hour to ensure smooth lines (not jagged)
    day_data = day_data.sort_values('hour').reset_index(drop=True)

    date_str = date.strftime('%B %d, %Y') if hasattr(date, 'strftime') else str(date)

    # EXACT layout from generate_visualizations.py: 4 panels vertical
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f'Catyra - Hot Day Analysis ({date_str})\n'
                 f'{n_towers} Towers × {tower_height}m Scale',
                 fontsize=14, fontweight='bold', y=0.98)

    hours = day_data['hour'].values

    # =========================================================================
    # Panel 1: Temperature Control
    # =========================================================================
    ax1 = axes[0]

    # Baseline (without cooling)
    ax1.plot(hours, day_data['T_baseline'],
             color=COLORS['baseline'], linestyle='--', linewidth=2,
             label='Without Cooling')

    # With cooling
    ax1.plot(hours, day_data['T_crop'],
             color=COLORS['cooling'], linestyle='-', linewidth=2,
             label='With Towers')

    # Fill between to show cooling effect
    ax1.fill_between(hours, day_data['T_baseline'], day_data['T_crop'],
                     alpha=0.3, color=COLORS['cooling_light'])

    # 95% confidence band on baseline (from RF model)
    if 'T_inside_lo' in day_data.columns and day_data['T_inside_lo'].notna().any():
        ax1.fill_between(hours, day_data['T_inside_lo'], day_data['T_inside_hi'],
                         alpha=0.12, color=COLORS['baseline'], label='95% CI (RF)')

    # Reference lines
    ax1.axhline(y=32, color='orange', linestyle=':', linewidth=1.5,
                label='Heat Stress (32°C)')
    ax1.axhline(y=35, color='red', linestyle=':', linewidth=1.5,
                label='Emergency (35°C)')

    # Optimal zone
    ax1.axhspan(22, 28, alpha=0.15, color='green', label='Optimal Zone')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_ylim(0, 45)
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax1.set_title('Temperature Control', fontweight='bold', loc='left')

    # Add cooling annotation
    max_cooling = (day_data['T_baseline'] - day_data['T_crop']).max()
    peak_idx = (day_data['T_baseline'] - day_data['T_crop']).argmax()
    peak_hour = hours[peak_idx]
    ax1.annotate(f'Max cooling: {max_cooling:.1f}°C',
                 xy=(peak_hour, day_data['T_crop'].iloc[peak_idx]),
                 xytext=(peak_hour + 2, day_data['T_crop'].iloc[peak_idx] - 3),
                 fontsize=9, color=COLORS['cooling'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['cooling'], lw=1.5))

    # =========================================================================
    # Panel 2: VD Control
    # =========================================================================
    ax2 = axes[1]

    # Danger zone background
    ax2.axhspan(12, 30, alpha=0.15, color='red', label='Damage Zone (>12)')
    ax2.axhspan(10, 12, alpha=0.15, color='orange', label='Stress Zone (10-12)')
    ax2.axhspan(4, 8, alpha=0.15, color='green', label='Optimal Zone (4-8)')

    # Baseline VD
    ax2.plot(hours, day_data['VD_baseline'],
             color=COLORS['baseline'], linestyle='--', linewidth=2,
             label='Without Cooling')

    # With cooling VD
    ax2.plot(hours, day_data['VD_crop'],
             color=COLORS['cooling'], linestyle='-', linewidth=2,
             label='With Towers')

    # Fill between
    ax2.fill_between(hours, day_data['VD_baseline'], day_data['VD_crop'],
                     alpha=0.3, color=COLORS['cooling_light'])

    # 95% confidence band on baseline VD (from RF model)
    if 'VD_inside_lo' in day_data.columns and day_data['VD_inside_lo'].notna().any():
        ax2.fill_between(hours, day_data['VD_inside_lo'], day_data['VD_inside_hi'],
                         alpha=0.12, color=COLORS['baseline'], label='95% CI (RF)')

    # Threshold lines
    ax2.axhline(y=10, color='orange', linestyle=':', linewidth=1.5)
    ax2.axhline(y=12, color='red', linestyle=':', linewidth=1.5)
    ax2.axhline(y=8, color='green', linestyle=':', linewidth=1, alpha=0.7)

    ax2.set_ylabel('Vapor Deficit (g/m³)')
    ax2.set_ylim(0, 35)
    ax2.set_xlim(0, 23)
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax2.set_title('VD Control', fontweight='bold', loc='left')

    # Add VD reduction annotation
    max_vd_reduction = (day_data['VD_baseline'] - day_data['VD_crop']).max()
    ax2.annotate(f'Max VD reduction: {max_vd_reduction:.1f} g/m³',
                 xy=(14, day_data['VD_crop'].iloc[14] if len(day_data) > 14 else 10),
                 xytext=(16, 5),
                 fontsize=9, color=COLORS['cooling'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['cooling'], lw=1.5))

    # =========================================================================
    # Panel 3: Humidity Control
    # =========================================================================
    ax3 = axes[2]

    # Baseline RH
    ax3.plot(hours, day_data['RH_baseline'],
             color=COLORS['baseline'], linestyle='--', linewidth=2,
             label='Without Cooling')

    # With cooling RH
    ax3.plot(hours, day_data['RH_crop'],
             color=COLORS['cooling'], linestyle='-', linewidth=2,
             label='With Towers')

    # Fill between
    ax3.fill_between(hours, day_data['RH_baseline'], day_data['RH_crop'],
                     where=day_data['RH_crop'] > day_data['RH_baseline'],
                     alpha=0.3, color=COLORS['cooling_light'])

    # Optimal zone
    ax3.axhspan(60, 80, alpha=0.15, color='green', label='Optimal Zone')

    # Reference lines
    ax3.axhline(y=85, color='red', linestyle=':', linewidth=1.5,
                label='Max RH (85%)')
    ax3.axhline(y=40, color='orange', linestyle=':', linewidth=1.5,
                label='Low RH Warning')

    ax3.set_ylabel('Relative Humidity (%)')
    ax3.set_ylim(20, 100)
    ax3.set_xlim(0, 23)
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax3.set_title('Humidity Control', fontweight='bold', loc='left')

    # =========================================================================
    # Panel 4: Control Timeline (with secondary axis for roof opening)
    # =========================================================================
    ax4 = axes[3]

    # Tower operation (binary) — left axis
    tower_on = day_data['tower_on'].astype(int) * 100
    ax4.fill_between(hours, 0, tower_on,
                     alpha=0.35, color=COLORS['tower_on'],
                     label='Towers ON', step='mid')

    # Stomatal opening (if available) — left axis
    if 'f_stomata' in day_data.columns:
        ax4.plot(hours, day_data['f_stomata'] * 100,
                 color='green', linestyle='--', linewidth=2,
                 label='Stomatal Opening')

    # Damage score (if available) — left axis
    if 'damage_score' in day_data.columns:
        ax4.plot(hours, day_data['damage_score'],
                 color='red', linestyle='-', linewidth=2,
                 label='Damage Score')

    ax4.set_ylabel('Tower / Stomata / Damage (%)')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylim(0, 110)
    ax4.set_xlim(0, 23)
    ax4.set_xticks(range(0, 24, 2))

    # Roof vent opening on SECONDARY y-axis (0-12%)
    if 'roof_pct' in day_data.columns and day_data['roof_pct'].max() > 0:
        ax4r = ax4.twinx()
        ax4r.plot(hours, day_data['roof_pct'],
                  color=COLORS['roof_open'], linestyle='-', linewidth=2.5,
                  label='Roof Opening (%)')
        ax4r.set_ylabel('Roof Opening (%)', color=COLORS['roof_open'])
        ax4r.set_ylim(0, max(15, day_data['roof_pct'].max() * 1.3))
        ax4r.tick_params(axis='y', labelcolor=COLORS['roof_open'])
        ax4r.legend(loc='upper right', framealpha=0.9)

    ax4.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax4.set_title('Control Timeline & Plant Response', fontweight='bold', loc='left')

    # Add summary statistics box
    tower_hours = day_data['tower_on'].sum()
    total_water = day_data['evap_total'].sum() / 1000
    damage_points = day_data['damage_towers'].sum() if 'damage_towers' in day_data.columns else 0

    stats_text = (f'Tower Runtime: {tower_hours} hrs\n'
                  f'Water Usage: {total_water:.1f} m³\n'
                  f'Damage Points: {damage_points:.2f}')

    ax4.text(0.98, 0.05, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.15)
    return fig


def create_annual_performance_3panel(results_df: pd.DataFrame, n_towers: int, tower_height: float) -> plt.Figure:
    """
    Create 3-panel annual performance plot - EXACT MATCH to generate_visualizations.py

    Panels:
    1. Daily Maximum VD
    2. Daily Maximum Temperature
    3. Daily Damage Score
    """
    # Daily aggregation
    daily = results_df.groupby('date').agg({
        'VD_baseline': 'max',
        'VD_crop': 'max',
        'T_baseline': 'max',
        'T_crop': 'max',
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    # Create figure - EXACT layout from generate_visualizations.py
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Catyra - Annual Performance Analysis\n'
                 f'{n_towers} Towers × {tower_height}m Scale | {daily["date"].min().strftime("%b %Y")} - {daily["date"].max().strftime("%b %Y")}',
                 fontsize=14, fontweight='bold', y=0.98)

    dates = daily['date']

    # =========================================================================
    # Panel 1: Daily Maximum VD
    # =========================================================================
    ax1 = axes[0]

    # Baseline fill (without cooling)
    ax1.fill_between(dates, 0, daily['VD_baseline'],
                     alpha=0.4, color=COLORS['baseline_light'],
                     label='Without Cooling')
    ax1.plot(dates, daily['VD_baseline'],
             color=COLORS['baseline'], linewidth=0.8, alpha=0.7)

    # With towers fill (overlapping)
    ax1.fill_between(dates, 0, daily['VD_crop'],
                     alpha=0.6, color=COLORS['cooling'],
                     label='With Towers')

    # Threshold lines
    ax1.axhline(y=10, color='orange', linestyle='--', linewidth=1.5,
                label='Stress Threshold (10 g/m³)')
    ax1.axhline(y=12, color='red', linestyle='--', linewidth=1.5,
                label='Damage Threshold (12 g/m³)')
    ax1.axhline(y=8, color='green', linestyle=':', linewidth=1, alpha=0.7)

    ax1.set_ylabel('Daily Max VD (g/m³)')
    ax1.set_ylim(0, max(daily['VD_baseline'].max() * 1.1, 20))
    ax1.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax1.set_title('Daily Maximum Vapor Deficit', fontweight='bold', loc='left')

    # Calculate VPD reduction
    vd_reduction_pct = (1 - daily['VD_crop'].mean() / daily['VD_baseline'].mean()) * 100
    ax1.text(0.02, 0.95, f'Avg. VD Reduction: {vd_reduction_pct:.0f}%',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', color=COLORS['cooling'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # Panel 2: Daily Maximum Temperature
    # =========================================================================
    ax2 = axes[1]

    # Baseline fill
    ax2.fill_between(dates, 0, daily['T_baseline'],
                     alpha=0.4, color=COLORS['baseline_light'],
                     label='Without Cooling')
    ax2.plot(dates, daily['T_baseline'],
             color=COLORS['baseline'], linewidth=0.8, alpha=0.7)

    # With towers fill
    ax2.fill_between(dates, 0, daily['T_crop'],
                     alpha=0.6, color=COLORS['cooling'],
                     label='With Towers')

    # Threshold lines
    ax2.axhline(y=32, color='orange', linestyle='--', linewidth=1.5,
                label='Heat Stress (32°C)')
    ax2.axhline(y=35, color='red', linestyle='--', linewidth=1.5,
                label='Emergency (35°C)')

    ax2.set_ylabel('Daily Max Temperature (°C)')
    ax2.set_ylim(10, max(daily['T_baseline'].max() * 1.1, 40))
    ax2.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax2.set_title('Daily Maximum Temperature', fontweight='bold', loc='left')

    # Calculate cooling
    avg_cooling = (daily['T_baseline'] - daily['T_crop']).mean()
    ax2.text(0.02, 0.95, f'Avg. Cooling: {avg_cooling:.1f}°C',
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', color=COLORS['cooling'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # Panel 3: Daily Damage Score
    # =========================================================================
    ax3 = axes[2]

    # Daily damage scores
    ax3.fill_between(dates, 0, daily['damage_baseline'],
                     alpha=0.4, color=COLORS['baseline_light'],
                     label='Est. Without Cooling')
    ax3.plot(dates, daily['damage_baseline'],
             color=COLORS['baseline'], linewidth=0.8, alpha=0.7)

    # With towers damage
    ax3.fill_between(dates, 0, daily['damage_towers'],
                     alpha=0.6, color=COLORS['cooling'],
                     label='With Towers')
    ax3.plot(dates, daily['damage_towers'],
             color=COLORS['cooling'], linewidth=0.8)

    ax3.set_ylabel('Daily Damage Points')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, max(daily['damage_baseline'].max() * 1.2,
                        daily['damage_towers'].max() * 1.5, 5))
    ax3.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax3.set_title('Daily Production Damage', fontweight='bold', loc='left')

    # Calculate damage reduction
    total_damage_baseline = daily['damage_baseline'].sum()
    total_damage_towers = daily['damage_towers'].sum()
    if total_damage_baseline > 0:
        damage_reduction_pct = (1 - total_damage_towers / total_damage_baseline) * 100
    else:
        damage_reduction_pct = 100 if total_damage_towers == 0 else 0

    ax3.text(0.02, 0.95,
             f'Total Damage: {total_damage_towers:.1f} pts (vs {total_damage_baseline:.1f} baseline)\n'
             f'Damage Reduction: {damage_reduction_pct:.0f}%',
             transform=ax3.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', color=COLORS['cooling'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.12)
    return fig


def create_damage_seasonal(results_df: pd.DataFrame, n_towers: int) -> plt.Figure:
    """Create seasonal damage analysis - exact match to generate_production_analysis.py"""
    results_df['month'] = pd.to_datetime(results_df['date']).dt.month

    monthly = results_df.groupby('month').agg({
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Catyra - Seasonal Damage Analysis\n{n_towers} Towers × 3.2m Scale',
                 fontsize=14, fontweight='bold', y=0.98)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = sorted(monthly['month'].unique())

    # Panel 1: Monthly damage bars
    ax1 = axes[0]
    x = np.arange(len(months))
    width = 0.35

    baseline_vals = [monthly[monthly['month'] == m]['damage_baseline'].values[0] for m in months]
    towers_vals = [monthly[monthly['month'] == m]['damage_towers'].values[0] for m in months]

    bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Without Cooling', color=COLORS['baseline'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, towers_vals, width, label='With Towers', color=COLORS['with_towers'], alpha=0.8)

    ax1.set_ylabel('Monthly Damage Points')
    ax1.set_xticks(x)
    ax1.set_xticklabels([month_names[m-1] for m in months])
    ax1.legend(loc='upper left')
    ax1.set_title('Monthly Damage Points', fontweight='bold', loc='left')

    for bar, val in zip(bars1, baseline_vals):
        if val > 5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8, color=COLORS['baseline'])
    for bar, val in zip(bars2, towers_vals):
        if val > 0.5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, color=COLORS['with_towers'])

    total_baseline = sum(baseline_vals)
    total_towers = sum(towers_vals)
    reduction_pct = (1 - total_towers / total_baseline) * 100 if total_baseline > 0 else 0

    ax1.text(0.98, 0.95, f'Total: {total_baseline:.0f} → {total_towers:.1f} pts\nReduction: {reduction_pct:.0f}%',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Panel 2: Cumulative damage
    ax2 = axes[1]
    daily = results_df.groupby('date').agg({
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily['cum_baseline'] = daily['damage_baseline'].cumsum()
    daily['cum_towers'] = daily['damage_towers'].cumsum()

    ax2.fill_between(daily['date'], 0, daily['cum_baseline'], alpha=0.3, color=COLORS['baseline_light'])
    ax2.plot(daily['date'], daily['cum_baseline'], color=COLORS['baseline'], linestyle='--', linewidth=2, label='Without Cooling')
    ax2.fill_between(daily['date'], 0, daily['cum_towers'], alpha=0.5, color=COLORS['with_towers'])
    ax2.plot(daily['date'], daily['cum_towers'], color=COLORS['with_towers'], linestyle='-', linewidth=2, label='With Towers')

    ax2.set_ylabel('Cumulative Damage Points')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.set_title('Cumulative Damage Over Season', fontweight='bold', loc='left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    final_baseline = daily['cum_baseline'].iloc[-1]
    final_towers = daily['cum_towers'].iloc[-1]
    ax2.annotate(f'{final_baseline:.0f} pts', xy=(daily['date'].iloc[-1], final_baseline),
                 xytext=(10, 0), textcoords='offset points', fontsize=10, color=COLORS['baseline'], fontweight='bold')
    ax2.annotate(f'{final_towers:.1f} pts', xy=(daily['date'].iloc[-1], final_towers),
                 xytext=(10, 0), textcoords='offset points', fontsize=10, color=COLORS['with_towers'], fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_production_cumulative(results_df: pd.DataFrame, prod_df: Optional[pd.DataFrame],
                                  economic: EconomicConfig, n_towers: int) -> plt.Figure:
    """Create production analysis - exact match to generate_production_analysis.py"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f'Catyra - Strawberry Production Analysis\nCaTaVa 2025 Season | {n_towers} Towers × 3.2m Scale',
                 fontsize=14, fontweight='bold', y=0.98)

    daily = results_df.groupby('date').agg({
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily['damage_prevented'] = daily['damage_baseline'] - daily['damage_towers']
    daily['cum_damage_prevented'] = daily['damage_prevented'].cumsum()

    plants_per_ha = 10000 * economic.plants_per_m2
    daily['production_gain_kg'] = daily['cum_damage_prevented'] * economic.damage_coeff * plants_per_ha / 1000

    if prod_df is not None and len(prod_df) > 0:
        ax.plot(prod_df['date'], prod_df['cumulative_kg'] * plants_per_ha,
                color=COLORS['baseline'], linestyle='-', linewidth=2.5,
                marker='o', markersize=4, label='Actual 2025 Production')

        prod_with_gain = prod_df.copy()
        prod_with_gain['projected_kg'] = prod_with_gain['cumulative_kg'] * plants_per_ha

        for idx, row in prod_with_gain.iterrows():
            date = row['date']
            damage_to_date = daily[daily['date'] <= date]['damage_prevented'].sum()
            gain = damage_to_date * economic.damage_coeff * plants_per_ha / 1000
            prod_with_gain.loc[idx, 'projected_kg'] = row['cumulative_kg'] * plants_per_ha + gain

        ax.plot(prod_with_gain['date'], prod_with_gain['projected_kg'],
                color=COLORS['with_towers'], linestyle='-', linewidth=2.5,
                marker='s', markersize=4, label='Projected With Towers')

        ax.fill_between(prod_with_gain['date'],
                        prod_with_gain['cumulative_kg'] * plants_per_ha,
                        prod_with_gain['projected_kg'],
                        alpha=0.3, color=COLORS['optimal_zone'], label='Production Gain Zone')

        final_actual = prod_with_gain['cumulative_kg'].iloc[-1] * plants_per_ha
        final_projected = prod_with_gain['projected_kg'].iloc[-1]
        total_gain = final_projected - final_actual
        gain_pct = (total_gain / final_actual) * 100 if final_actual > 0 else 0
        revenue_gain = total_gain * economic.price_per_kg

        stats_text = (f'Actual Production: {final_actual:,.0f} kg/ha\n'
                      f'Projected with Towers: {final_projected:,.0f} kg/ha\n'
                      f'Production Gain: {total_gain:,.0f} kg/ha (+{gain_pct:.1f}%)\n'
                      f'Revenue Gain: €{revenue_gain:,.0f}/ha')
    else:
        total_prevented = daily['damage_prevented'].sum()
        prod_gain = total_prevented * economic.damage_coeff * plants_per_ha / 1000
        revenue = prod_gain * economic.price_per_kg

        ax.plot(daily['date'], daily['production_gain_kg'],
                color=COLORS['optimal_zone'], linestyle='-', linewidth=2.5, label='Cumulative Production Gain')
        ax.fill_between(daily['date'], 0, daily['production_gain_kg'],
                        alpha=0.3, color=COLORS['optimal_zone'])

        stats_text = (f'Total Damage Prevented: {total_prevented:,.0f} pts\n'
                      f'Production Gain: {prod_gain:,.0f} kg/ha\n'
                      f'Revenue Gain: €{revenue:,.0f}/ha')

    ax.set_ylabel('Cumulative Production (kg/ha)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.set_title('Cumulative Strawberry Production', fontweight='bold', loc='left')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_temperature_annual(results_df: pd.DataFrame, n_towers: int) -> plt.Figure:
    """Create temperature analysis - exact match to generate_production_analysis.py"""
    results_df['day_period'] = results_df['hour'].apply(lambda h: 'day' if 6 <= h <= 20 else 'night')

    day_df = results_df[results_df['day_period'] == 'day']
    night_df = results_df[results_df['day_period'] == 'night']

    daily_day = day_df.groupby('date').agg({'T_baseline': 'mean', 'T_crop': 'mean'}).reset_index()
    daily_day.columns = ['date', 'T_day_baseline', 'T_day_towers']
    daily_day['date'] = pd.to_datetime(daily_day['date'])

    daily_night = night_df.groupby('date').agg({'T_baseline': 'mean', 'T_crop': 'mean'}).reset_index()
    daily_night.columns = ['date', 'T_night_baseline', 'T_night_towers']
    daily_night['date'] = pd.to_datetime(daily_night['date'])

    daily_avg = results_df.groupby('date').agg({'T_baseline': 'mean', 'T_crop': 'mean'}).reset_index()
    daily_avg.columns = ['date', 'T_avg_baseline', 'T_avg_towers']
    daily_avg['date'] = pd.to_datetime(daily_avg['date'])

    merged = daily_avg.merge(daily_day, on='date', how='left').merge(daily_night, on='date', how='left')
    merged = merged.sort_values('date')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Catyra - Temperature Analysis\n{n_towers} Towers × 3.2m Scale',
                 fontsize=14, fontweight='bold', y=0.98)

    dates = merged['date']

    # Panel 1: Daily Average
    ax1 = axes[0]
    ax1.fill_between(dates, 0, merged['T_avg_baseline'], alpha=0.3, color=COLORS['baseline_light'])
    ax1.plot(dates, merged['T_avg_baseline'], color=COLORS['baseline'], linestyle='--', linewidth=1.5, label='Without Cooling')
    ax1.fill_between(dates, 0, merged['T_avg_towers'], alpha=0.5, color=COLORS['with_towers'])
    ax1.plot(dates, merged['T_avg_towers'], color=COLORS['with_towers'], linestyle='-', linewidth=1.5, label='With Towers')
    ax1.axhline(y=20, color=COLORS['optimal_zone'], linestyle=':', linewidth=1.5, label='Optimal (20°C)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right', ncol=3)
    ax1.set_title('Daily Average Temperature', fontweight='bold', loc='left')
    ax1.set_ylim(5, 35)

    # Panel 2: Day Temperature
    ax2 = axes[1]
    ax2.fill_between(dates, 0, merged['T_day_baseline'], alpha=0.3, color=COLORS['baseline_light'])
    ax2.plot(dates, merged['T_day_baseline'], color=COLORS['baseline'], linestyle='--', linewidth=1.5, label='Without Cooling')
    ax2.fill_between(dates, 0, merged['T_day_towers'], alpha=0.5, color=COLORS['with_towers'])
    ax2.plot(dates, merged['T_day_towers'], color=COLORS['with_towers'], linestyle='-', linewidth=1.5, label='With Towers')
    ax2.axhline(y=20, color=COLORS['optimal_zone'], linestyle=':', linewidth=1.5, label='Optimal Day (20°C)')
    ax2.axhline(y=28, color=COLORS['stress'], linestyle=':', linewidth=1.5, label='Stress (28°C)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc='upper right', ncol=4)
    ax2.set_title('Day Temperature (6:00-20:00)', fontweight='bold', loc='left')
    ax2.set_ylim(5, 40)

    # Panel 3: Night Temperature
    ax3 = axes[2]
    ax3.fill_between(dates, 0, merged['T_night_baseline'], alpha=0.5, color=COLORS['production'])
    ax3.plot(dates, merged['T_night_baseline'], color=COLORS['production'], linestyle='-', linewidth=1.5, label='Night Temperature')
    ax3.axhline(y=14, color=COLORS['optimal_zone'], linestyle=':', linewidth=1.5, label='Optimal Night (14°C)')
    ax3.axhline(y=18, color=COLORS['stress'], linestyle=':', linewidth=1.5, label='High Night (18°C)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper right', ncol=3)
    ax3.set_title('Night Temperature (20:00-6:00)', fontweight='bold', loc='left')
    ax3.set_ylim(5, 30)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.15)
    return fig


def create_dif_analysis(results_df: pd.DataFrame, n_towers: int) -> plt.Figure:
    """Create DIF analysis - exact match to generate_production_analysis.py"""
    results_df['day_period'] = results_df['hour'].apply(lambda h: 'day' if 6 <= h <= 20 else 'night')

    day_df = results_df[results_df['day_period'] == 'day']
    night_df = results_df[results_df['day_period'] == 'night']

    day_temps = day_df.groupby('date')['T_baseline'].mean().reset_index()
    night_temps = night_df.groupby('date')['T_baseline'].mean().reset_index()

    dif_df = day_temps.merge(night_temps, on='date', suffixes=('_day', '_night'))
    dif_df['date'] = pd.to_datetime(dif_df['date'])
    dif_df['DIF_baseline'] = dif_df['T_baseline_day'] - dif_df['T_baseline_night']

    # Calculate improved DIF with cooling
    daily_cooling = results_df.groupby('date')['cooling'].max().reset_index()
    daily_cooling['date'] = pd.to_datetime(daily_cooling['date'])  # Fix: ensure same dtype
    dif_df = dif_df.merge(daily_cooling, on='date')
    dif_df['DIF_improved'] = dif_df['DIF_baseline'] - dif_df['cooling'] * 0.5

    dif_df['month'] = dif_df['date'].dt.month
    monthly_dif = dif_df.groupby('month').agg({'DIF_baseline': 'mean', 'DIF_improved': 'mean'}).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Catyra - DIF Analysis (Day-Night Temperature Difference)\n{n_towers} Towers × 3.2m Scale',
                 fontsize=14, fontweight='bold', y=0.98)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = sorted(dif_df['month'].unique())

    # Panel 1: Monthly DIF
    ax1 = axes[0]
    x = np.arange(len(months))
    width = 0.35

    baseline_vals = [monthly_dif[monthly_dif['month'] == m]['DIF_baseline'].values[0] for m in months]
    improved_vals = [monthly_dif[monthly_dif['month'] == m]['DIF_improved'].values[0] for m in months]

    ax1.bar(x - width/2, baseline_vals, width, label='Without Cooling', color=COLORS['baseline'], alpha=0.8)
    ax1.bar(x + width/2, improved_vals, width, label='With Towers', color=COLORS['with_towers'], alpha=0.8)
    ax1.axhline(y=4.7, color=COLORS['optimal_zone'], linestyle='--', linewidth=2, label='Min Optimal DIF (4.7°C)')
    ax1.axhline(y=8, color=COLORS['optimal_zone'], linestyle=':', linewidth=1.5, label='Max Optimal DIF (8°C)')

    ax1.set_ylabel('DIF (°C)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([month_names[m-1] for m in months])
    ax1.legend(loc='upper left', ncol=2)
    ax1.set_title('Monthly Average DIF', fontweight='bold', loc='left')
    ax1.set_ylim(0, max(baseline_vals) * 1.2)

    # Panel 2: Daily DIF
    ax2 = axes[1]
    dif_sorted = dif_df.sort_values('date')

    ax2.fill_between(dif_sorted['date'], 0, dif_sorted['DIF_baseline'], alpha=0.3, color=COLORS['baseline_light'])
    ax2.plot(dif_sorted['date'], dif_sorted['DIF_baseline'], color=COLORS['baseline'], linestyle='--', linewidth=1.5, label='Without Cooling')
    ax2.fill_between(dif_sorted['date'], 0, dif_sorted['DIF_improved'], alpha=0.5, color=COLORS['with_towers'])
    ax2.plot(dif_sorted['date'], dif_sorted['DIF_improved'], color=COLORS['with_towers'], linestyle='-', linewidth=1.5, label='With Towers')
    ax2.axhspan(4.7, 8, alpha=0.15, color=COLORS['optimal_zone'], label='Optimal DIF Zone')
    ax2.axhline(y=4.7, color=COLORS['optimal_zone'], linestyle='--', linewidth=1.5)

    ax2.set_ylabel('DIF (°C)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', ncol=3)
    ax2.set_title('Daily DIF Over Season', fontweight='bold', loc='left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.2)
    return fig


def create_economic_summary(results_df: pd.DataFrame, economic: EconomicConfig, n_towers: int) -> plt.Figure:
    """Create economic summary - exact match to generate_production_analysis.py"""
    results_df['month'] = pd.to_datetime(results_df['date']).dt.month

    daily = results_df.groupby('date').agg({
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()
    daily['damage_prevented'] = daily['damage_baseline'] - daily['damage_towers']

    plants_per_ha = 10000 * economic.plants_per_m2
    total_baseline = daily['damage_baseline'].sum()
    total_towers = daily['damage_towers'].sum()
    total_prevented = daily['damage_prevented'].sum()

    production_gain_kg = total_prevented * economic.damage_coeff * plants_per_ha / 1000
    revenue_gain = production_gain_kg * economic.price_per_kg

    monthly = results_df.groupby('month').agg({
        'damage_baseline': 'sum',
        'damage_towers': 'sum',
    }).reset_index()
    monthly['damage_prevented'] = monthly['damage_baseline'] - monthly['damage_towers']
    monthly['production_gain_kg'] = monthly['damage_prevented'] * economic.damage_coeff * plants_per_ha / 1000
    monthly['revenue_gain'] = monthly['production_gain_kg'] * economic.price_per_kg

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Catyra - Economic Impact Analysis\n{n_towers} Towers × 3.2m Scale | 1 Hectare Greenhouse',
                 fontsize=14, fontweight='bold', y=0.98)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = sorted(monthly['month'].unique())
    x = np.arange(len(months))

    # Panel 1: Damage prevented
    ax1 = axes[0, 0]
    prevented = [monthly[monthly['month'] == m]['damage_prevented'].values[0] for m in months]
    ax1.bar(x, prevented, color=COLORS['with_towers'], alpha=0.8)
    ax1.set_ylabel('Damage Points Prevented')
    ax1.set_xticks(x)
    ax1.set_xticklabels([month_names[m-1] for m in months])
    ax1.set_title('Monthly Damage Prevention', fontweight='bold', loc='left')

    # Panel 2: Production gain
    ax2 = axes[0, 1]
    prod = [monthly[monthly['month'] == m]['production_gain_kg'].values[0] for m in months]
    ax2.bar(x, prod, color=COLORS['optimal_zone'], alpha=0.8)
    ax2.set_ylabel('Production Gain (kg/ha)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([month_names[m-1] for m in months])
    ax2.set_title('Monthly Production Gain', fontweight='bold', loc='left')

    # Panel 3: Revenue
    ax3 = axes[1, 0]
    rev = [monthly[monthly['month'] == m]['revenue_gain'].values[0] for m in months]
    ax3.bar(x, rev, color=COLORS['production'], alpha=0.8)
    ax3.set_ylabel('Revenue Gain (€/ha)')
    ax3.set_xlabel('Month')
    ax3.set_xticks(x)
    ax3.set_xticklabels([month_names[m-1] for m in months])
    ax3.set_title('Monthly Revenue Gain', fontweight='bold', loc='left')

    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    reduction_pct = (total_prevented / total_baseline * 100) if total_baseline > 0 else 0

    summary = f"""
╔══════════════════════════════════════════════════╗
║        SEASON ECONOMIC SUMMARY                   ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  DAMAGE ANALYSIS                                 ║
║  ─────────────────────────────────────────────   ║
║  Baseline Damage:      {total_baseline:>10,.0f} pts          ║
║  With Towers:          {total_towers:>10,.1f} pts          ║
║  Damage Prevented:     {total_prevented:>10,.0f} pts          ║
║  Reduction:            {reduction_pct:>10.0f} %            ║
║                                                  ║
║  PRODUCTION IMPACT                               ║
║  ─────────────────────────────────────────────   ║
║  Damage Coefficient:   {economic.damage_coeff:>10.3f} g/pt         ║
║  Plants per Hectare:   {int(plants_per_ha):>10,}              ║
║  Production Gain:      {production_gain_kg:>10,.0f} kg/ha        ║
║                                                  ║
║  ECONOMIC VALUE                                  ║
║  ─────────────────────────────────────────────   ║
║  Price per kg:         €{economic.price_per_kg:>9.2f}              ║
║  Revenue Gain:         €{revenue_gain:>9,.0f}/ha          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""

    ax4.text(0.1, 0.95, summary, transform=ax4.transAxes, fontsize=11, fontfamily='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f9ff', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.markdown('<div class="main-header">🌡️ Catyra - Climate Simulator</div>',
                unsafe_allow_html=True)

    if USE_FULL_MODEL:
        st.success("✅ Using calibrated model (greenhouse_climate_v2.py)")
    else:
        st.warning("⚠️ Using simplified model. Place greenhouse_climate_v2.py in the webapp folder for accurate results.")

    # Initialize session state
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'climate_df' not in st.session_state:
        st.session_state.climate_df = None
    if 'production_df' not in st.session_state:
        st.session_state.production_df = None
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None
    if 'historical_df' not in st.session_state:
        st.session_state.historical_df = None
    if 'historical_scenario' not in st.session_state:
        st.session_state.historical_scenario = None

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 📁 Data Upload")
        climate_file = st.file_uploader("Climate Data", type=['xlsx', 'csv'],
            help="Excel/CSV with: datetime, T_inside, RH_inside")
        production_file = st.file_uploader("Production Data (optional)", type=['xlsx', 'csv'],
            help="CaTaVa production data for yield analysis")
        forecast_file = st.file_uploader("Weather Forecast (7-day CSV)", type=['csv'],
            help="CSV from HTML weather tool (semicolon-separated)",
            key="forecast_upload")
        historical_file = st.file_uploader("Historical Data (multi-year CSV)", type=['csv'],
            help="Multi-year CSV from HTML weather tool with RF corrections",
            key="historical_upload")

        st.markdown("---")

        st.markdown("### 🌿 Environment")
        env_type = st.radio("Model type", ["🏠 Greenhouse", "🌳 Orchard"], horizontal=True, key="env_type")
        is_greenhouse = env_type.startswith("🏠")

        if is_greenhouse:
            st.markdown("### 🏠 Greenhouse Geometry")
            col1, col2 = st.columns(2)
            with col1:
                gh_area = st.number_input("Floor Area (m²)", value=10000, min_value=1000, max_value=100000, step=1000)
                gh_h_crop = st.number_input("Crop Height (m)", value=1.5, min_value=0.5, max_value=3.0, step=0.1)
            with col2:
                gh_h_roof = st.number_input("Roof Height (m)", value=4.2, min_value=3.0, max_value=8.0, step=0.1)

            st.info(f"Area: {gh_area:,} m² ({gh_area/10000:.1f} ha)")

            greenhouse = GreenhouseConfig(A_floor=gh_area, H_crop=gh_h_crop, H_roof=gh_h_roof) if USE_FULL_MODEL else \
                type('GH', (), {'A_floor': gh_area, 'H_crop': gh_h_crop, 'H_roof': gh_h_roof, 'volume': gh_area * gh_h_roof, 'RH_max': 85})()
            orchard = None
        else:
            st.markdown("### 🌳 Orchard Layout")
            col1, col2 = st.columns(2)
            with col1:
                towers_per_ha = st.number_input("Towers per hectare", value=25, min_value=5, max_value=60, step=1)
            with col2:
                row_spacing = st.number_input("Row spacing (m)", value=3.5, min_value=2.0, max_value=6.0, step=0.5)

            st.info(f"Density: {towers_per_ha} towers/ha · {row_spacing}m rows")

            orchard = OrchardConfig(towers_per_ha=towers_per_ha, row_spacing=row_spacing) if USE_FULL_MODEL else \
                type('Orch', (), {'towers_per_ha': towers_per_ha, 'row_spacing': row_spacing})()
            greenhouse = GreenhouseConfig() if USE_FULL_MODEL else \
                type('GH', (), {'A_floor': 10000, 'H_crop': 1.5, 'H_roof': 4.2, 'volume': 42000, 'RH_max': 85})()
            gh_area = 10000  # not used but needed for fallback

        st.markdown("---")

        st.markdown("### 🗼 Tower Configuration")
        if is_greenhouse:
            n_towers = st.slider("Number of Towers", min_value=10, max_value=60, value=32, step=2)
        else:
            n_towers = towers_per_ha  # orchard uses towers_per_ha directly
        col1, col2 = st.columns(2)
        with col1:
            tower_height = st.number_input("Tower Height (m)", value=3.2, min_value=2.0, max_value=5.0, step=0.1)
            tower_c = st.number_input("C_discharge", value=0.61, min_value=0.3, max_value=0.8, step=0.01)
        with col2:
            tower_d = st.number_input("Diameter (m)", value=1.0, min_value=0.5, max_value=2.0, step=0.1)
            tower_evap = st.number_input("Max Evap (L/hr)", value=25, min_value=10, max_value=50)
        tower_venturi = st.number_input("Venturi / keeldiameter (m)", value=0.95, min_value=0.30, max_value=2.0, step=0.05,
                                        help="Bepaalt de keelsnelheid (verstuiving/menging), niet het debiet. "
                                             "Het debiet volgt uit de torendoorsnede — data-onderbouwd op 5 schaalmodel-reeksen.")

        tower = TowerConfig(H_tower=tower_height, D_top=tower_d, D_bottom=tower_d,
                           C_discharge=tower_c, V_dot_evap_max=tower_evap) if USE_FULL_MODEL else \
            type('Tower', (), {'H_tower': tower_height, 'D_top': tower_d, 'D_bottom': tower_d,
                              'C_discharge': tower_c, 'V_dot_evap_max': tower_evap, 'H_discharge': 0.5,
                              'W_supply': 100})()
        # Venturi als attribuut zetten (werkt ook als een oudere engine dit veld nog niet als
        # constructor-argument kent; dan wordt het simpelweg genegeerd tot de engine is bijgewerkt).
        try:
            tower.D_venturi = tower_venturi
        except Exception:
            pass

        st.markdown("---")

        st.markdown("### 🎛️ Control Parameters")
        col1, col2 = st.columns(2)
        with col1:
            vd_on = st.number_input("VD ON (g/m³)", value=8.0, min_value=6.0, max_value=12.0, step=0.5)
            hour_start = st.number_input("Start Hour", value=6, min_value=0, max_value=12)
        with col2:
            vd_off = st.number_input("VD OFF (g/m³)", value=6.0, min_value=4.0, max_value=10.0, step=0.5)
            hour_end = st.number_input("End Hour", value=20, min_value=12, max_value=23)

        control = ControlConfig(VD_tower_on=vd_on, VD_tower_off=vd_off,
                               hour_start=hour_start, hour_end=hour_end) if USE_FULL_MODEL else \
            type('Control', (), {'VD_tower_on': vd_on, 'VD_tower_off': vd_off,
                                'hour_start': hour_start, 'hour_end': hour_end})()

        st.markdown("---")

        st.markdown("### 💰 Economic Parameters")
        col1, col2 = st.columns(2)
        with col1:
            damage_coeff = st.number_input("Damage Coeff (g/pt)", value=0.083, min_value=0.01, max_value=0.2, step=0.001, format="%.3f")
            price_kg = st.number_input("Price (€/kg)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
        with col2:
            plants_m2 = st.number_input("Plants/m²", value=6.0, min_value=2.0, max_value=12.0, step=0.5)

        economic = EconomicConfig(damage_coeff=damage_coeff, plants_per_m2=plants_m2, price_per_kg=price_kg)

        with st.expander("🔬 Advanced Parameters"):
            mixing_model = st.radio(
                "Mixing Model",
                ["Webapp7 (calibrated)", "GreenLight (validated)"],
                index=0,
                help="Webapp7: single eta_mix=0.23, strat=0.75\nGreenLight: eta_T=0.34, eta_RH=0.54, strat=0.85"
            )

            if mixing_model == "GreenLight (validated)":
                strat = st.slider("Stratification", 0.5, 1.0, 0.85, 0.05)
                ach = st.slider("ACH Reference", 5, 20, 10)
                eta = st.slider("Mixing Efficiency (base)", 0.1, 0.4, 0.23, 0.01)
                eta_temp = st.slider("Eta Temperature", 0.1, 0.6, 0.34, 0.01)
                eta_rh = st.slider("Eta Humidity", 0.1, 0.8, 0.54, 0.01)
                hf = st.slider("Height Factor", 0.5, 0.9, 0.70, 0.05)
            else:
                strat = st.slider("Stratification", 0.5, 1.0, 0.75, 0.05)
                ach = st.slider("ACH Reference", 5, 20, 10)
                eta = st.slider("Mixing Efficiency", 0.1, 0.4, 0.23, 0.01)
                eta_temp = None
                eta_rh = None
                hf = st.slider("Height Factor", 0.5, 0.9, 0.70, 0.05)

            vd_optimal = st.number_input("VD Optimal", value=8.0)
            vd_stress = st.number_input("VD Stress", value=10.0)

        if USE_FULL_MODEL:
            mixing = MixingConfig(stratification=strat, ACH_ref=ach, eta_mix=eta,
                                  height_factor=hf, eta_temp=eta_temp, eta_rh=eta_rh)
        else:
            mixing = type('Mixing', (), {'stratification': strat, 'ACH_ref': ach, 'eta_mix': eta,
                                         'height_factor': hf, 'eta_temp': eta_temp, 'eta_rh': eta_rh})()
        crop = CropConfig(VD_optimal=vd_optimal, VD_stress=vd_stress) if USE_FULL_MODEL else \
            type('Crop', (), {'VD_optimal': vd_optimal, 'VD_stress': vd_stress, 'VD_critical': 12.0, 'VD_lethal': 15.0})()

        st.markdown("---")
        run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    # ==========================================================================
    # LOAD DATA (before tabs so both tabs can access it)
    # ==========================================================================

    if climate_file:
        if st.session_state.climate_df is None or climate_file.name not in str(st.session_state.get('climate_file_name', '')):
            with st.spinner("Loading climate data..."):
                st.session_state.climate_df = load_climate_data(climate_file)
                st.session_state.climate_file_name = climate_file.name

    if production_file:
        if st.session_state.production_df is None or production_file.name not in str(st.session_state.get('prod_file_name', '')):
            with st.spinner("Loading production data..."):
                st.session_state.production_df = load_production_data(production_file)
                st.session_state.prod_file_name = production_file.name

    if historical_file:
        if st.session_state.historical_df is None or historical_file.name not in str(st.session_state.get('historical_file_name', '')):
            with st.spinner("Loading historical data..."):
                st.session_state.historical_df = _try_load_historical_csv(historical_file)
                st.session_state.historical_file_name = historical_file.name

    if forecast_file:
        if st.session_state.forecast_df is None or forecast_file.name not in str(st.session_state.get('forecast_file_name', '')):
            with st.spinner("Loading forecast data..."):
                fc_df = _try_load_forecast_csv(forecast_file)
                if fc_df is None:
                    forecast_file.seek(0)
                    fc_df = load_climate_data(forecast_file)
                st.session_state.forecast_df = fc_df
                st.session_state.forecast_file_name = forecast_file.name
                for k in ['fc_results_low', 'fc_results_medium', 'fc_results_high']:
                    st.session_state[k] = None

    # ==========================================================================
    # TOP-LEVEL TABS: Climate Year vs Forecast
    # ==========================================================================
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Jaaranalyse", "🔮 Weerbericht", "🗼 Toren-performance"])

    # ==========================================================================
    # TAB 1: JAARANALYSE (Full-year climate analysis)
    # ==========================================================================
    with main_tab1:
        if st.session_state.climate_df is not None:
            df = st.session_state.climate_df
            prod_df = st.session_state.production_df

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Days", f"{df['date'].nunique()}")
            with col2:
                st.metric("⏱️ Hours", f"{len(df):,}")
            with col3:
                st.metric("🌡️ Max Temp", f"{df['T_inside'].max():.1f}°C")
            with col4:
                e_sat = 0.61121 * np.exp((18.678 - df['T_inside'] / 234.5) * (df['T_inside'] / (257.14 + df['T_inside'])))
                e_act = e_sat * df['RH_inside'] / 100
                vd_calc = (e_sat - e_act) * 1000 / (461.5 * (df['T_inside'] + 273.15)) * 1000
                st.metric("💨 Max VD", f"{vd_calc.max():.1f} g/m³")

            if run_btn:
                with st.spinner("Running simulation..."):
                    st.session_state.results_df = run_simulation(df, n_towers, tower, greenhouse, control, crop, mixing,
                                                                       is_greenhouse=is_greenhouse, orchard=orchard)
                    st.success("✅ Simulation complete!")

            if st.session_state.results_df is not None:
                results = st.session_state.results_df

                total_baseline = results['damage_baseline'].sum()
                total_towers = results['damage_towers'].sum()
                damage_prevented = total_baseline - total_towers
                reduction_pct = (damage_prevented / total_baseline * 100) if total_baseline > 0 else 0
                plants_ha = 10000 * economic.plants_per_m2
                prod_gain = damage_prevented * economic.damage_coeff * plants_ha / 1000
                revenue = prod_gain * economic.price_per_kg
                water_m3 = results['evap_total'].sum() / 1000

                st.markdown("---")
                st.markdown("### 📊 Key Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Damage Baseline", f"{total_baseline:,.0f} pts")
                with col2:
                    st.metric("With Towers", f"{total_towers:,.1f} pts", f"-{reduction_pct:.1f}%")
                with col3:
                    st.metric("Production Gain", f"{prod_gain:,.0f} kg/ha")
                with col4:
                    st.metric("Revenue Gain", f"€{revenue:,.0f}/ha")
                with col5:
                    st.metric("Water Used", f"{water_m3:,.0f} m³")

                st.markdown("---")
                tabs = st.tabs(["🌡️ Hot Day", "📈 Annual", "🎯 Damage", "🌾 Production",
                               "🌡️ Temperature", "📊 DIF", "💰 Economic", "📋 Export"])

                with tabs[0]:
                    st.markdown("### Hot Day 4-Panel Analysis")
                    dates = sorted(results['date'].unique())
                    daily_max_vd = results.groupby('date')['VD_baseline'].max()
                    hottest = daily_max_vd.idxmax()
                    selected = st.selectbox("Select Date", dates, index=dates.index(hottest) if hottest in dates else 0,
                        format_func=lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
                    fig = create_hot_day_4panel(results, selected, n_towers, tower_height)
                    if fig:
                        st.pyplot(fig)
                        plt.close()

                with tabs[1]:
                    st.markdown("### Annual Performance 3-Panel")
                    fig = create_annual_performance_3panel(results, n_towers, tower_height)
                    st.pyplot(fig)
                    plt.close()

                with tabs[2]:
                    st.markdown("### Seasonal Damage Analysis")
                    fig = create_damage_seasonal(results, n_towers)
                    st.pyplot(fig)
                    plt.close()

                with tabs[3]:
                    st.markdown("### Production Analysis")
                    fig = create_production_cumulative(results, prod_df, economic, n_towers)
                    st.pyplot(fig)
                    plt.close()

                with tabs[4]:
                    st.markdown("### Temperature Analysis")
                    fig = create_temperature_annual(results, n_towers)
                    st.pyplot(fig)
                    plt.close()

                with tabs[5]:
                    st.markdown("### DIF Analysis")
                    fig = create_dif_analysis(results, n_towers)
                    st.pyplot(fig)
                    plt.close()

                with tabs[6]:
                    st.markdown("### Economic Summary")
                    fig = create_economic_summary(results, economic, n_towers)
                    st.pyplot(fig)
                    plt.close()

                with tabs[7]:
                    st.markdown("### Data Export")
                    st.dataframe(results.head(100), width='stretch')
                    csv = results.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "simulation_results.csv", "text/csv")

        else:
            st.markdown("""
            <div class="info-box">
            <h3>📊 Jaaranalyse</h3>
            <p>Upload klimaatdata (jaar) in de sidebar om de volledige analyse te starten.</p>
            <p><strong>Vereist:</strong> datetime, T_inside, RH_inside</p>
            <p><strong>Optioneel:</strong> Productiedata voor opbrengstanalyse</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🎮 Load Demo Data"):
                for path in [Path(__file__).parent.parent / 'greenhousedata1.xlsx', Path(__file__).parent / 'demo_data.xlsx']:
                    if path.exists():
                        try:
                            demo = pd.read_excel(path, header=None, skiprows=3)
                            demo.columns = ['datetime', 'VD_measured', 'T_inside', 'RH_inside', 'T_outside', 'radiation', 'wind_speed', 'radiation_out'][:len(demo.columns)]
                            demo['datetime'] = pd.to_datetime(demo['datetime'], errors='coerce')
                            demo['hour'] = demo['datetime'].dt.hour
                            demo['date'] = demo['datetime'].dt.date
                            for c in ['T_inside', 'RH_inside']:
                                demo[c] = pd.to_numeric(demo[c], errors='coerce')
                            demo['RH_outside'] = (demo['RH_inside'] - 10).clip(25, 90)
                            demo = demo.dropna(subset=['datetime', 'T_inside', 'RH_inside'])
                            st.session_state.climate_df = demo
                            st.session_state.climate_file_name = 'demo'
                            st.success("✅ Demo loaded!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                        break

        # ------------------------------------------------------------------
        # HISTORICAL 10-YEAR ANALYSIS (shown regardless of climate data)
        # ------------------------------------------------------------------
        if st.session_state.historical_df is not None:
            st.markdown("---")
            st.markdown("### 📅 Historische VD Analyse (10 jaar)")

            hist_df = st.session_state.historical_df
            years = sorted(hist_df['year'].unique())
            st.info(f"Historische data: {len(years)} jaar ({int(min(years))}–{int(max(years))}) · {len(hist_df):,} uurwaarden")

            # Nozzle type selector
            nozzle_type = st.radio(
                "Nozzle type",
                ["Klein (1.76 L/hr)", "Groot (2.70 L/hr)"],
                horizontal=True, key="hist_nozzle_type"
            )
            nozzle_flow = 1.76 if "Klein" in nozzle_type else 2.70

            hist_stats = compute_historical_vd_stats(
                hist_df, H_tower=tower_height, C_discharge=tower_c, D_bottom=tower_d
            )

            # Key metrics row
            worst = hist_stats['worst_year']
            best = hist_stats['best_year']
            pcts = hist_stats['daily_percentiles']
            evap_pcts = hist_stats['evap_percentiles']

            # Peak evaporation at P95
            p95_peak_evap = evap_pcts['P95'].max()
            n_nozzles_p95 = int(np.ceil(p95_peak_evap / nozzle_flow))
            water_supply_p95 = n_nozzles_p95 * nozzle_flow  # L/hr total supply needed

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Slechtste jaar", f"{worst['year']}", f"peak {worst['peak_vd']:.1f} g/m³")
            with col2:
                st.metric("Beste jaar", f"{best['year']}", f"peak {best['peak_vd']:.1f} g/m³")
            with col3:
                st.metric("P95 max VD", f"{pcts['P95'].max():.1f} g/m³")
            with col4:
                st.metric("P95 verdamping", f"{p95_peak_evap:.1f} L/hr",
                          help="Maximale verdamping per toren bij P95 condities")
            with col5:
                st.metric("Nozzles (P95)", f"{n_nozzles_p95}",
                          f"{water_supply_p95:.0f} L/hr water",
                          help=f"Aantal nozzles nodig bij {nozzle_flow} L/hr per nozzle")
            with col6:
                p99_evap = evap_pcts['P99'].max()
                n_nozzles_p99 = int(np.ceil(p99_evap / nozzle_flow))
                st.metric("Nozzles (P99)", f"{n_nozzles_p99}",
                          f"{p99_evap:.1f} L/hr",
                          help="Nozzles voor extreme condities (1% overschrijding)")

            # 4-panel chart
            fig = create_historical_vd_chart(hist_stats, nozzle_flow=nozzle_flow)
            st.pyplot(fig)
            plt.close()

            # Detailed tables
            with st.expander("📊 Maandelijkse percentiel tabel (VD + verdamping)"):
                # VD percentiles by month
                pcts_month = pcts.copy()
                pcts_month['month'] = pd.to_datetime(pcts_month['day_of_year'], format='%j', errors='coerce').dt.month
                monthly_vd = pcts_month.groupby('month')[['P50', 'P75', 'P90', 'P95', 'P99']].max().round(1)

                # Evaporation percentiles by month
                evap_month = evap_pcts.copy()
                evap_month['month'] = pd.to_datetime(evap_month['day_of_year'], format='%j', errors='coerce').dt.month
                monthly_evap = evap_month.groupby('month')[['P50', 'P75', 'P90', 'P95', 'P99']].max().round(1)

                # Nozzle count from P95 evaporation
                monthly_nozzles = (monthly_evap / nozzle_flow).apply(np.ceil).astype(int)

                month_labels = ['Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt']

                st.markdown("**Max VD per dag (g/m³)**")
                monthly_vd.index = month_labels[:len(monthly_vd)]
                st.dataframe(monthly_vd, use_container_width=True)

                st.markdown("**Verdamping per toren (L/hr)**")
                monthly_evap.index = month_labels[:len(monthly_evap)]
                st.dataframe(monthly_evap, use_container_width=True)

                st.markdown(f"**Aantal nozzles nodig ({nozzle_flow} L/hr per stuk)**")
                monthly_nozzles.index = month_labels[:len(monthly_nozzles)]
                st.dataframe(monthly_nozzles, use_container_width=True)

                st.markdown(f"""
                **Leeswijzer:**
                - P95 = in 95% van de jaren wordt deze waarde niet overschreden
                - De P95 verdamping bepaalt hoeveel nozzles je nodig hebt per toren
                - Toren: {tower_height}m, C={tower_c}, D={tower_d}m
                - Nozzle: {nozzle_type} @ {nozzle_flow} L/hr per stuk
                """)

    # ==========================================================================
    # TAB 2: WEERBERICHT (Weather Forecast)
    # ==========================================================================
    with main_tab2:
        if st.session_state.forecast_df is not None:
            fc_df = st.session_state.forecast_df

            fc_dates = sorted(fc_df['date'].unique())
            n_fc_days = len(fc_dates)

            # Check if RF confidence bands are available
            has_ci = 'T_inside_lo' in fc_df.columns and fc_df['T_inside_lo'].notna().any()

            # Scenario selector
            if has_ci:
                scenario = st.radio(
                    "RF prediction scenario",
                    ["Medium (best estimate)", "Low (optimistic — cooler)", "High (worst case — hotter)"],
                    horizontal=True, key="fc_scenario"
                )
                scenario_key = scenario.split(" ")[0].lower()
                st.info(f"Forecast loaded: {n_fc_days} days ({fc_dates[0]} to {fc_dates[-1]}) · Scenario: **{scenario_key}**")
            else:
                scenario_key = "medium"
                st.info(f"Forecast loaded: {n_fc_days} days ({fc_dates[0]} to {fc_dates[-1]})")

            # Prepare forecast df based on scenario
            fc_sim_df = fc_df.copy()
            if has_ci and scenario_key != "medium":
                suffix = '_lo' if scenario_key == 'low' else '_hi'
                # Swap T_inside with the selected scenario
                if f'T_inside{suffix}' in fc_sim_df.columns:
                    fc_sim_df['T_inside'] = fc_sim_df[f'T_inside{suffix}'].combine_first(fc_sim_df['T_inside'])
                # Back-calculate RH from T and VD to keep physics consistent
                if f'VD_inside{suffix}' in fc_sim_df.columns:
                    vd_target = fc_sim_df[f'VD_inside{suffix}'].combine_first(fc_sim_df.get('VD_measured', pd.Series(dtype=float)))
                    T_new = fc_sim_df['T_inside']
                    e_sat = 0.61121 * np.exp((18.678 - T_new / 234.5) * (T_new / (257.14 + T_new)))
                    e_sat_pa = e_sat * 1000
                    Tk = T_new + 273.15
                    Mw = 18.015e-3
                    R = 8.314
                    e_act = e_sat_pa - (vd_target / 1000) * R * Tk / Mw
                    rh_new = np.clip(100 * e_act / e_sat_pa, 15, 99)
                    mask = vd_target.notna()
                    fc_sim_df.loc[mask, 'RH_inside'] = rh_new[mask].round(1)

            # Cache key based on scenario + all model settings
            settings_hash = (f"{scenario_key}_{is_greenhouse}_{n_towers}_{tower.H_tower}_{tower.D_bottom}_{tower.C_discharge}_{tower.V_dot_evap_max}_{getattr(tower, 'D_venturi', 0.95)}"
                             f"_{control.VD_tower_on}_{control.VD_tower_off}_{control.hour_start}_{control.hour_end}"
                             f"_{mixing.stratification}_{mixing.ACH_ref}_{mixing.eta_mix}_{mixing.height_factor}_{mixing.eta_temp}_{mixing.eta_rh}"
                             f"_{crop.VD_optimal}_{crop.VD_stress}")
            if not is_greenhouse and orchard:
                settings_hash += f"_{orchard.towers_per_ha}_{orchard.row_spacing}"
            fc_cache_key = f"fc_results_{settings_hash}"
            # Clear old caches if settings changed
            old_keys = [k for k in st.session_state if k.startswith('fc_results_') and k != fc_cache_key]
            for k in old_keys:
                del st.session_state[k]
            if fc_cache_key not in st.session_state or st.session_state.get(fc_cache_key) is None:
                with st.spinner(f"Running forecast simulation ({scenario_key})..."):
                    st.session_state[fc_cache_key] = run_simulation(
                        fc_sim_df, n_towers, tower, greenhouse, control, crop, mixing,
                        is_greenhouse=is_greenhouse, orchard=orchard
                    )

            fc_results = st.session_state[fc_cache_key]

            if fc_results is not None:
                # Day navigator
                fc_result_dates = sorted(fc_results['date'].unique())

                # Summary metrics per day
                daily_summary = []
                for d in fc_result_dates:
                    day_data = fc_results[fc_results['date'] == d]
                    daily_summary.append({
                        'date': d,
                        'max_T': day_data['T_baseline'].max(),
                        'max_VD': day_data['VD_baseline'].max(),
                        'max_VD_towers': day_data['VD_crop'].max(),
                        'tower_hrs': day_data['tower_on'].sum(),
                        'damage_base': day_data['damage_baseline'].sum(),
                        'damage_towers': day_data['damage_towers'].sum(),
                    })

                # Show day summary cards
                cols = st.columns(min(n_fc_days, 7))
                for i, ds in enumerate(daily_summary[:7]):
                    with cols[i]:
                        d = ds['date']
                        date_str = d.strftime('%a %d/%m') if hasattr(d, 'strftime') else str(d)
                        st.markdown(f"**{date_str}**")
                        st.caption(f"T: {ds['max_T']:.0f}°C | VD: {ds['max_VD']:.0f}→{ds['max_VD_towers']:.0f}")
                        st.caption(f"Tower: {ds['tower_hrs']:.0f}h | Dmg: {ds['damage_towers']:.1f}")

                # Day-by-day 4-panel charts
                st.markdown("---")
                selected_fc_date = st.selectbox(
                    "Select forecast day",
                    fc_result_dates,
                    format_func=lambda x: x.strftime('%A %d %B %Y') if hasattr(x, 'strftime') else str(x),
                    key="forecast_day_select"
                )

                fig = create_hot_day_4panel(fc_results, selected_fc_date, n_towers, tower_height)
                if fig:
                    st.pyplot(fig)
                    plt.close()

                # Button to re-run with updated settings
                if st.button("🔄 Re-run forecast simulation"):
                    for k in [f"fc_results_low", f"fc_results_medium", f"fc_results_high"]:
                        st.session_state[k] = None
                    st.rerun()

        else:
            st.markdown("""
            <div class="info-box">
            <h3>🔮 Weerbericht</h3>
            <p>Upload een 7-daags weerbericht CSV in de sidebar.</p>
            <p>Exporteer vanuit de <strong>HTML Weather Tool</strong> (semicolon-separated).</p>
            <p>De simulatie draait automatisch met dezelfde toren- en kasinstellingen.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: TOREN-PERFORMANCE (losse berekening — luchthoeveelheid per toren)
    # ==========================================================================
    with main_tab3:
        st.markdown("### 🗼 Toren-performance")
        st.caption("De **luchthoeveelheid per toren** is de belangrijkste prestatiemaat. Stel hieronder de "
                   "afmetingen, het aantal torens en de omgeving in — deze pagina rekent los van de simulatie, "
                   "dus je kunt vrij variëren zonder de sidebar te wijzigen.")

        st.markdown("#### ⚙️ Torenafmetingen")
        p1, p2, p3 = st.columns(3)
        with p1:
            perf_n = st.number_input("Aantal torens", min_value=1, max_value=200,
                                     value=int(n_towers), step=1, key="perf_n")
            perf_H = st.number_input("Hoogte (m)", min_value=1.0, max_value=8.0,
                                     value=float(tower_height), step=0.1, key="perf_H")
        with p2:
            perf_D = st.number_input("Diameter (m)", min_value=0.3, max_value=3.0,
                                     value=float(tower_d), step=0.1, key="perf_D")
            perf_nozzle = st.selectbox("Nozzle → C_discharge (gevalideerd)",
                                       ["Foggy — C = 0,61", "Gardena — C = 0,56", "Handmatig"],
                                       key="perf_nozzle",
                                       help="Gevalideerde afvoercoëfficiënt uit de Technical Briefing (dec 2025).")
        with p3:
            perf_evap = st.number_input("Max Evap (L/hr)", min_value=5.0, max_value=80.0,
                                        value=float(tower_evap), step=1.0, key="perf_evap")
            if perf_nozzle.startswith("Foggy"):
                perf_C = 0.61
            elif perf_nozzle.startswith("Gardena"):
                perf_C = 0.56
            else:
                perf_C = st.number_input("C_discharge (handmatig)", min_value=0.30, max_value=0.90,
                                         value=float(tower_c), step=0.01, key="perf_C")
        A_outlet = float(np.pi * (perf_D / 2) ** 2)
        st.caption(f"Uitlaatoppervlak  A = π·(D/2)² = **{A_outlet:.3f} m²**")

        st.markdown("#### 🌤️ Inblaascondities (omgeving)")
        oc1, oc2 = st.columns(2)
        with oc1:
            perf_T = st.slider("Buitentemperatuur (°C)", 15.0, 40.0, 30.0, 0.5, key="perf_T")
        with oc2:
            perf_RH = st.slider("Relatieve vochtigheid (%)", 20.0, 90.0, 50.0, 1.0, key="perf_RH")

        perf = calc_tower_perf(perf_T, perf_RH, H_tower=perf_H,
                               C_discharge=perf_C, D_bottom=perf_D, evap_max=perf_evap)
        Vdot_hr = float(perf['V_dot_hr'])
        Vdot_s = float(perf['V_dot_s'])
        Vout = float(perf['V_out'])
        total_hr = Vdot_hr * perf_n

        # ---- Belangrijkste maat: luchthoeveelheid per toren ----
        st.markdown("#### 💨 Luchthoeveelheid per toren")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Per toren", f"{Vdot_hr:,.0f} m³/hr", help="V̇ = V_uit × A_uitlaat × 3600")
        m2.metric("Per toren", f"{Vdot_s:,.2f} m³/s")
        m3.metric("Uitlaatsnelheid", f"{Vout:.2f} m/s")
        m4.metric(f"Totaal ({perf_n} torens)", f"{total_hr:,.0f} m³/hr")

        gh_vol = getattr(greenhouse, 'volume', greenhouse.A_floor * getattr(greenhouse, 'H_roof', 4.2))
        ach = total_hr / gh_vol if gh_vol > 0 else 0
        st.caption(f"Kasvolume ≈ {gh_vol:,.0f} m³ → **{ach:.2f} luchtwisselingen per uur** met alle torens samen.")

        # ---- Venturi (keel) ----
        st.markdown("#### 🌀 Venturi (keel)")
        perf_venturi = st.number_input("Venturi-/keeldiameter (m)", min_value=0.05, max_value=3.0,
                                       value=float(tower_venturi), step=0.01, key="perf_venturi",
                                       help="Zelfde parameter als in de sidebar.")
        A_keel = float(np.pi * (perf_venturi / 2) ** 2)
        ratio = perf_venturi / perf_D if perf_D > 0 else 0
        V_keel_cont = Vdot_s / A_keel if A_keel > 0 else 0.0      # continuïteit: keelsnelheid
        accel = V_keel_cont / Vout if Vout > 0 else 0
        st.caption(f"Keeloppervlak A_keel = π·(D_keel/2)² = **{A_keel:.3f} m²**  ·  keel/uitlaat = {ratio:.2f}×")
        st.caption("✅ **Data-onderbouwd (schaalmodel feb 2026):** de venturi versnelt de keelsnelheid maar bepaalt "
                   "**niet** het debiet. Het gemeten debiet volgt uit de torendoorsnede met C ≈ 0,61–0,66 (5 reeksen). "
                   "De simulatie rekent daarom het debiet op de torendoorsnede; de venturi dient voor de keelsnelheid "
                   "(fijnere verstuiving, betere menging).")

        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Debiet (op torendoorsnede)**")
            st.metric("Per toren", f"{Vdot_hr:,.0f} m³/hr")
            st.metric(f"Totaal ({perf_n} torens)", f"{Vdot_hr * perf_n:,.0f} m³/hr")
        with vc2:
            st.markdown("**Keelsnelheid (venturi)**")
            st.metric("v_keel", f"{V_keel_cont:.2f} m/s", help="V_keel = V̇ / A_keel (continuïteit)")
            st.metric("Versnelling", f"{accel:.2f}×", help="keelsnelheid / uitlaatsnelheid")

        # ---- Uitblaascondities ----
        st.markdown("#### 🌬️ Uitblaascondities (uit de toren)")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Luchthoeveelheid", f"{Vdot_hr:,.0f} m³/hr")
        u2.metric("Uitblaastemperatuur", f"{float(perf['T_out']):.1f} °C",
                  delta=f"{-float(perf['dT']):.1f} °C", delta_color="inverse",
                  help="Koelt af tot ~natteboltemperatuur")
        u3.metric("Uitblaas-luchtvochtigheid", f"{float(perf['RH_out']):.0f} %",
                  help="Lucht verlaat de toren (nagenoeg) verzadigd")
        u4.metric("Massastroom", f"{float(perf['m_dot']):.2f} kg/s")

        st.caption(
            f"Inblaas: {perf_T:.1f} °C / {perf_RH:.0f} %RV  →  "
            f"Uitblaas: {float(perf['T_out']):.1f} °C / {float(perf['RH_out']):.0f} %RV  "
            f"(ΔT = {float(perf['dT']):.1f} °C afkoeling)"
        )

        st.markdown("#### 🌡️ Overige prestatie")
        e1, e2, e3 = st.columns(3)
        e1.metric("Natteboltemp.", f"{float(perf['T_wb']):.1f} °C")
        e2.metric("Verdamping / toren", f"{float(perf['evap']):.1f} L/hr")
        e3.metric("Koelvermogen / toren", f"{float(perf['cooling_kW']):.1f} kW")

        # ---- Grafiek ----
        st.markdown("#### 📈 Luchthoeveelheid per toren vs. buitentemperatuur")
        T_range = np.linspace(15, 40, 60)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for rh in [30, 50, 70]:
            p = calc_tower_perf(T_range, np.full_like(T_range, rh),
                                H_tower=perf_H, C_discharge=perf_C,
                                D_bottom=perf_D, evap_max=perf_evap)
            ax.plot(T_range, p['V_dot_hr'], linewidth=2, label=f"RH {rh}%")
        ax.axvline(perf_T, color='gray', ls='--', alpha=0.6)
        ax.set_xlabel("Buitentemperatuur (°C)")
        ax.set_ylabel("Luchthoeveelheid per toren (m³/hr)")
        ax.set_title(f"{perf_H:.1f} m hoog · Ø {perf_D:.2f} m · C = {perf_C:.2f}")
        ax.legend(title="Rel. vocht")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # ---- Tabel ----
        st.markdown("#### 📋 Tabel (bij ingestelde relatieve vochtigheid)")
        T_tab = np.arange(20, 39, 2.0)
        p_tab = calc_tower_perf(T_tab, np.full_like(T_tab, perf_RH),
                                H_tower=perf_H, C_discharge=perf_C,
                                D_bottom=perf_D, evap_max=perf_evap)
        tbl = pd.DataFrame({
            "T inblaas (°C)": T_tab,
            "T uitblaas (°C)": np.round(p_tab['T_out'], 1),
            "RV uitblaas (%)": np.round(p_tab['RH_out'], 0),
            "V_uit (m/s)": np.round(p_tab['V_out'], 2),
            "Lucht/toren (m³/hr)": np.round(p_tab['V_dot_hr'], 0),
            f"Totaal {perf_n} torens (m³/hr)": np.round(p_tab['V_dot_hr'] * perf_n, 0),
            "Verdamping (L/hr)": np.round(p_tab['evap'], 1),
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        with st.expander("ℹ️ Gebruikte formules"):
            st.markdown(r"""
- **Uitlaatsnelheid (schoorsteeneffect):** $V_{uit} = C \cdot \sqrt{\dfrac{2\,g\,H\,\Delta T}{T_{amb,K}}}$
- **Uitlaatoppervlak:** $A = \pi (D/2)^2$
- **Luchthoeveelheid per toren:** $\dot V = V_{uit}\cdot A \cdot 3600$  [m³/hr]
- **ΔT** = inblaastemperatuur − uitblaastemperatuur; de uitblaas koelt af tot ~natteboltemperatuur (Stull).
- **Uitblaas-luchtvochtigheid** volgt uit de vochtopname (verdamping / luchtmassastroom); bij niet-begrensde
  verdamping is de uitblaas verzadigd (≈100 %RV).
- **Massastroom:** $\dot m = \rho_{lucht}\cdot V_{uit}\cdot A$, met $\rho_{lucht} = 1{,}2$ kg/m³.
- **Verdamping:** $\dot m \cdot \Delta w \cdot 3600$, begrensd op *Max Evap*.
- **Venturi — opvatting A (continuïteit):** debiet gelijk, keelsnelheid $V_{keel} = \dot V / A_{keel}$.
- **Venturi — opvatting B (keel bepaalt debiet):** $\dot V = V_{uit}\cdot A_{keel}$, met $A_{keel}=\pi(D_{keel}/2)^2$.

Exact dezelfde fysica als in `greenhouse_climate_v2.py`.
""")

        with st.expander("📐 Gevalideerde referentiedata (Technical Briefing, dec 2025)"):
            st.markdown("**Model vs. briefing — 3,5 m toren (Ø 1,2 m, Foggy C = 0,61)**")
            _cond = [("Dutch summer", 31, 40, 0.91, 21),
                     ("Hot greenhouse", 35, 20, 1.15, 19),
                     ("Desert (Sinai)", 45, 10, 1.40, 21)]
            _rows = []
            for nm, T, RH, Vb, Tb in _cond:
                _p = calc_tower_perf(T, RH, H_tower=3.5, C_discharge=0.61, D_bottom=1.2, evap_max=100)
                _rows.append({
                    "Conditie": nm, "T_in (°C)": T, "RH (%)": RH,
                    "V model (m/s)": round(float(_p['V_out']), 2), "V briefing (m/s)": Vb,
                    "T_out model (°C)": round(float(_p['T_out']), 1), "T_out briefing (°C)": Tb,
                    "Lucht model (m³/hr)": round(float(_p['V_dot_hr']), 0),
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            st.caption("Het model reproduceert de voorspelde uitlaatsnelheid binnen ~1 %.")

            st.markdown("**Gemeten 2 m-toren (Ø 0,68 m) — steady-state (>15 min)**")
            st.dataframe(pd.DataFrame([
                {"Test": "Longrun 1", "Nozzle": "4× Foggy", "T_in (°C)": "28–32", "V_out (m/s)": 0.72, "RH_out (%)": 97, "C": 0.61},
                {"Test": "Longrun 2", "Nozzle": "5× Foggy", "T_in (°C)": "27–30", "V_out (m/s)": 0.76, "RH_out (%)": 99, "C": 0.61},
                {"Test": "Gardena", "Nozzle": "5× Gardena", "T_in (°C)": "33–35", "V_out (m/s)": 0.75, "RH_out (%)": 97, "C": 0.56},
            ]), use_container_width=True, hide_index=True)

            st.markdown("""
**Kerncijfers uit de validatie**
- C_discharge = **0,61 ± 0,03** (Foggy, 75+ meetpunten, σ = 3 %) · **0,56** (Gardena)
- Uitlaat = natteboltemperatuur; verzadigingsrendement η = **97–100 %**
- Snelheidsschaling **V ∝ √h**: 1 m→2 m gemeten 1,44× (theorie 1,41×)
- Gevalideerd bereik: T_in 27–35 °C, RH 17–49 %
- 3,5 m-toren: H 3,5 m · Ø 1,2 m · **venturi 0,95 m** · capaciteit 23–25 kW @ 35 °C/20 %
- Spreiding/menging (aparte laag): η_mix = 0,23–0,34; zwaartekrachtstroom U = 0,7·√(g'·H)

Bron: *CatavaAir CoolFlow Technical Briefing*, december 2025.
""")

        with st.expander("🔬 Schaalmodel-validatie — 5 meetreeksen (Ø700 × 2000 mm, feb 2026)"):
            st.markdown("**Model vs. meting** — stack-model op de torendoorsnede (Ø700), C = 0,61.")
            g = 9.81
            A_meet = float(np.pi * (0.700 / 2) ** 2)   # torendoorsnede schaalmodel
            _series = [
                ("8R", 8, 0, 29.99, 7.88, 1050.1),
                ("4R", 4, 0, 29.72, 12.03, 803.5),
                ("4R+4G", 4, 4, 30.56, 9.37, 934.9),
                ("4G", 0, 4, 31.84, 10.72, 1104.3),
                ("8G", 0, 8, 29.12, 8.40, 1131.8),
            ]
            _rows = []
            for lbl, nr, ng, Tin, dT, Vmeas in _series:
                v = 0.61 * (2 * g * 2.0 * dT / (273.15 + Tin)) ** 0.5
                Vmodel = v * A_meet * 3600
                Ceff = (Vmeas / 3600 / A_meet) / ((2 * g * 2.0 * dT / (273.15 + Tin)) ** 0.5)
                _rows.append({
                    "Reeks": lbl, "Nozzles": f"{nr}R+{ng}G", "T_in (°C)": Tin, "ΔT (K)": dT,
                    "V gemeten (m³/h)": round(Vmeas),
                    "V model C=0,61 (m³/h)": round(Vmodel),
                    "Afwijking": f"{(Vmodel - Vmeas) / Vmeas * 100:+.0f}%",
                    "C effectief": round(Ceff, 3),
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            st.markdown("""
**Conclusie uit deze validatie**
- Het stack-model op de **torendoorsnede** reproduceert het gemeten debiet binnen de spreiding; gemiddelde effectieve **C = 0,66 ± 0,11**.
- De **venturi** (Ø500 keel) versnelt de keelsnelheid tot ~1,5 m/s, maar bepaalt niet het debiet — daarom rekent de simulatie op de torendoorsnede.
- **C stijgt met de nozzle-belasting** (0,42 bij 4R tot 0,73 bij 8G): het ejector-effect van de nozzles voegt luchtstroom toe. Dit is een reëel tweede-orde-effect dat het model als vaste C benadert.
- Water was in de meeste reeksen **ruim overvloedig** (injectie 4–18 L/h vs. verdampt 4–7 L/h) → verdamping is fysica-begrensd, zoals het model aanneemt.

Bron: *Catyra Air Torensimulator v4 — schaalmodelmetingen*, februari 2026.
""")

        with st.expander("💧 Nozzle-referentie — Ikeuchi KBN (q ∝ √P)"):
            st.markdown("Gevalideerde nozzle-flows uit de Ikeuchi-datasheet. Debiet schaalt met de wortel van de druk: "
                        "q(P) = q₁₀ · √(P / 10 bar).")
            _bars = [3, 5, 7, 10, 15, 20]
            _noz = {"KBN 80125 (rood)": 4.10, "KBN 80063 (groen)": 2.00}
            _tab = {"Druk (bar)": _bars}
            for name, qref in _noz.items():
                _tab[name + " (L/h)"] = [round(qref * (b / 10) ** 0.5, 2) for b in _bars]
            st.dataframe(pd.DataFrame(_tab), use_container_width=True, hide_index=True)
            st.caption("KBN 80125 = 4,10 L/h @ 10 bar · KBN 80063 = 2,00 L/h @ 10 bar. "
                       "Nozzle-aantal en -type beïnvloeden de prestatie vooral via het ejector-effect op de luchtstroom "
                       "(hogere C), niet via de watertoevoer — die is doorgaans niet de beperkende factor.")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;font-size:0.8rem;'>Catyra v2.0 | © 2026 Catyra Engineering</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
