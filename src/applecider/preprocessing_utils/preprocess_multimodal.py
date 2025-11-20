#!/usr/bin/env python3
import os, io, gzip, json, math, warnings, inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.time import Time  # only used if you later want ISO->MJD for spectra (we usually want that)

warnings.filterwarnings('ignore', category=AstropyWarning)

# ----------------------------
# Optional dependencies
# ----------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

try:
    from scipy.interpolate import interp1d
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    interp1d = None
    scipy_stats = None

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_dir: Path
    spec_csv: Path
    output_root: Path

    # Photometry -> merge
    delta_t_hours: float = 12.0

    # Alert matching
    alert_tol_days: float = 0.5
    require_all_3_cuts: bool = True
    allow_fallback_nearest_any: bool = True
    max_nearest_any_dt_days: Optional[float] = None  # None = unlimited

    # Spectra
    spectrum_wave_min: float = 4500.0
    spectrum_wave_max: float = 7980.0
    spectrum_step: float = 1.0  # angstroms

    # Misc
    random_seed: int = 42

    def wave_grid(self) -> np.ndarray:
        n = int(round((self.spectrum_wave_max - self.spectrum_wave_min) / self.spectrum_step)) + 1
        return np.linspace(self.spectrum_wave_min, self.spectrum_wave_max, n, dtype=np.float32)

# ----------------------------
# Constants / maps
# ----------------------------
FIDS = [1, 2, 3]
FID2BAND = {1: 'ztfg', 2: 'ztfr', 3: 'ztfi'}
BAND2ID  = {'ztfg': 0, 'ztfr': 1, 'ztfi': 2}
LOG_CONST = 1.0/np.log(10)

# ----------------------------
# Merge kernel
# ----------------------------
if NUMBA_OK:
    @njit
    def _merge_jit(time, flux, err, dt_days, eps=1e-8):
        n = time.shape[0]
        times_out = np.empty(n, np.float64)
        fluxes_out = np.empty(n, np.float64)
        errs_out = np.empty(n, np.float64)
        cnt = 0
        i = 0
        while i < n:
            t0 = time[i]
            j = i
            while j + 1 < n and time[j + 1] - t0 <= dt_days:
                j += 1
            totw = 0.0
            for k in range(i, j + 1):
                totw += 1.0/(err[k] + eps)
            tw = fw = ew = 0.0
            for k in range(i, j + 1):
                w = (1.0/(err[k] + eps))/totw
                tw += w * time[k]
                fw += w * flux[k]
                ew += w * err[k]
            times_out[cnt]  = tw
            fluxes_out[cnt] = fw
            errs_out[cnt]   = ew
            cnt += 1
            i = j + 1
        return times_out[:cnt], fluxes_out[:cnt], errs_out[:cnt]
else:
    def _merge_jit(time, flux, err, dt_days, eps=1e-8):
        times_out, fluxes_out, errs_out = [], [], []
        i = 0; n = len(time)
        while i < n:
            t0 = time[i]
            j = i
            while j + 1 < n and time[j + 1] - t0 <= dt_days:
                j += 1
            w = (1.0/(err[i:j+1] + eps)); w = w/w.sum()
            times_out.append(np.sum(w * time[i:j+1]))
            fluxes_out.append(np.sum(w * flux[i:j+1]))
            errs_out.append(np.sum(w * err[i:j+1]))
            i = j + 1
        return np.asarray(times_out), np.asarray(fluxes_out), np.asarray(errs_out)

# ----------------------------
# helpers
# ----------------------------
def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if SCIPY_OK:
        try:
            return float(scipy_stats.median_abs_deviation(x, scale=1.0, nan_policy='omit'))
        except Exception:
            pass
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _interp_with_extrap(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xnew = np.asarray(xnew, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.full_like(xnew, np.nan, dtype=np.float64)
    if SCIPY_OK and interp1d is not None:
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=True)
        return f(xnew)
    ynew = np.interp(xnew, x, y)
    left = xnew < x[0]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        ynew[left] = y[0] + slope * (xnew[left] - x[0])
    right = xnew > x[-1]
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        ynew[right] = y[-1] + slope * (xnew[right] - x[-1])
    return ynew

# ----------------------------
# Photometry utils
# ----------------------------
def mag_to_flux(mag, magerr):
    """ for taming spectra """
    flux = 10 ** (-0.4 * (mag - 23.9))
    flux_err = (magerr / (2.5 / np.log(10))) * flux
    return flux, flux_err

def _normalize_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ for standardizing columns between alerts.npy and photometry.csv """
    df = df.copy()
    if 'fid' in df.columns and df['fid'].notna().any():
        df['fid'] = pd.to_numeric(df['fid'], errors='coerce').astype('Int64')
        df['filter'] = df.get('filter', pd.Series([pd.NA]*len(df))).copy()
        df.loc[df['filter'].isna(), 'filter'] = df.loc[df['filter'].isna(), 'fid'].map(FID2BAND)
    elif 'filter' in df.columns and df['filter'].notna().any():
        mapping = {'ztfg': 'ztfg', 'ztfr': 'ztfr', 'ztfi': 'ztfi', 'g': 'ztfg', 'r': 'ztfr', 'i': 'ztfi'}
        df['filter'] = df['filter'].astype(str).str.strip().str.lower().map(mapping)
        inv = {'ztfg': 1, 'ztfr': 2, 'ztfi': 3}
        df['fid'] = df['filter'].map(inv).astype('Int64')
    else:
        df['filter'] = np.nan
        df['fid'] = pd.Series([pd.NA]*len(df), dtype='Int64')
    return df

def _read_csv_photometry(obj_id: str, data_dir: Path) -> pd.DataFrame:
    p = data_dir/obj_id/'photometry.csv'
    if not p.exists():
        return pd.DataFrame(columns=['obj_id','jd','mjd','mag','magerr','fid','filter'])
    df = pd.read_csv(p)
    df = df.rename(columns={'magpsf':'mag', 'sigmapsf':'magerr', 'jdobs':'jd', 'MJD':'mjd', 'JD':'jd'})
    if 'jd' not in df.columns and 'mjd' in df.columns:
        df['jd'] = df['mjd'] + 2400000.5
    if 'mjd' not in df.columns and 'jd' in df.columns:
        df['mjd'] = df['jd'] - 2400000.5
    df = df.dropna(subset=['jd','mjd','mag','magerr']).copy()
    df = _normalize_filter_columns(df)
    df['flux'], df['flux_error'] = mag_to_flux(df['mag'].astype(float), df['magerr'].astype(float))
    df['obj_id'] = obj_id
    keep = ['obj_id','jd','mjd','mag','magerr','flux','flux_error','fid','filter']
    df = df[keep]
    df = df[df['filter'].isin(['ztfg','ztfr','ztfi'])]
    return df.reset_index(drop=True)

def _read_alert_candidate_photometry(obj_id: str, data_dir: Path) -> pd.DataFrame:
    a = data_dir/obj_id/'alerts.npy'
    if not a.exists():
        return pd.DataFrame(columns=['obj_id','jd','mjd','mag','magerr','fid','filter'])
    arr = np.load(a, allow_pickle=True)
    alerts = list(arr) if isinstance(arr, np.ndarray) else arr
    rows = []
    for al in alerts:
        c = al.get('candidate', {})
        try:
            jd  = float(c['jd'])
            mag = float(c.get('magpsf', np.nan))
            me  = float(c.get('sigmapsf', np.nan))
            fid = int(c.get('fid', 0))
        except Exception:
            continue
        if not np.isfinite([jd, mag, me]).all() or fid not in (1,2,3):
            continue
        rows.append({'obj_id': obj_id, 'jd': jd, 'mjd': jd-2400000.5,
                     'mag': mag, 'magerr': me, 'fid': fid,
                     'filter': FID2BAND.get(fid, np.nan)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df['flux'], df['flux_error'] = mag_to_flux(df['mag'].astype(float), df['magerr'].astype(float))
    return df.reset_index(drop=True)

def _dedup_pref_csv(unified: pd.DataFrame, jd_round_decimals: int = 5) -> pd.DataFrame:
    """ deduplicate (?) light curve observations that are shared by alerts.npy and photometry.csv """
    if 'source' not in unified.columns:
        unified['source'] = 'unknown'
    unified = unified.sort_values(['source']).copy()
    rounded = unified.copy()
    rounded['jd_round'] = rounded['jd'].round(jd_round_decimals)
    rounded = rounded.drop_duplicates(subset=['fid','jd_round'], keep='first')
    rounded = rounded.drop(columns=['jd_round'])
    return rounded.reset_index(drop=True)

def load_photometry_obj(obj_id: str, data_dir: Path) -> pd.DataFrame:
    csv_df = _read_csv_photometry(obj_id, data_dir)
    if len(csv_df):
        csv_df['source'] = 'csv'
    al_df  = _read_alert_candidate_photometry(obj_id, data_dir)
    if len(al_df):
        al_df['source'] = 'alerts'
    if len(csv_df)==0 and len(al_df)==0:
        return pd.DataFrame(columns=['obj_id','jd','mjd','mag','magerr','flux','flux_error','fid','filter'])
    uni = pd.concat([csv_df, al_df], ignore_index=True)
    uni = uni[uni['filter'].isin(['ztfg','ztfr','ztfi'])].copy()
    if len(uni) == 0:
        return uni
    uni = _dedup_pref_csv(uni, jd_round_decimals=5)
    mjd0 = uni['mjd'].min()
    uni['mjd'] = uni['mjd'] - mjd0
    keep = ['obj_id','jd','mjd','mag','magerr','flux','flux_error','fid','filter']
    return uni[keep].reset_index(drop=True)

def merge_by_filter(df: pd.DataFrame, delta_t_hours=12.0) -> pd.DataFrame:
    out = []
    for band, grp in df.groupby('filter'):
        if band not in BAND2ID:
            continue
        grp2 = grp.sort_values('mjd')
        if len(grp2) == 0:
            continue
        t_arr = grp2['mjd'].to_numpy()
        f_arr = grp2['flux'].to_numpy()
        e_arr = grp2['flux_error'].to_numpy()
        dt_days = delta_t_hours/24.0
        t_out, f_out, e_out = _merge_jit(t_arr, f_arr, e_arr, dt_days)
        m = pd.DataFrame({'mjd': t_out, 'flux': f_out, 'flux_error': e_out})
        m['jd']     = m['mjd'] + (grp2['jd'].min() - grp2['mjd'].min())
        m['filter'] = band
        m['fid']    = {'ztfg':1,'ztfr':2,'ztfi':3}[band]
        m['obj_id'] = grp2['obj_id'].iloc[0]
        out.append(m)
    if not out:
        return pd.DataFrame(columns=['mjd','flux','flux_error','jd','filter','fid','obj_id'])
    return pd.concat(out, ignore_index=True)

def build_event_features(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.sort_values('mjd').reset_index(drop=True)
    if len(df) == 0:
        return df
    t0 = df['mjd'].iloc[0]
    dt = df['mjd'] - t0
    dt_prev = np.diff(np.r_[t0, df['mjd'].values])
    f = np.clip(df['flux'].values.astype(np.float32), 1e-6, None)
    logf = np.log10(f)
    sig_logf = df['flux_error'].values.astype(np.float32) * LOG_CONST / f
    out = pd.DataFrame({
        'dt': dt.values.astype(np.float32),
        'dt_prev': dt_prev.astype(np.float32),
        'band_id': df['filter'].map(BAND2ID).astype(np.int8),
        'logflux': logf.astype(np.float32),
        'logflux_err': sig_logf.astype(np.float32),
        'jd': df['jd'].values.astype(np.float64),
        'fid': df['fid'].values.astype(np.int16),
        'obj_id': df['obj_id'].values
    })
    for band, idx in BAND2ID.items():
        out[f'band_{band}'] = (out['band_id']==idx).astype(np.float32)
    TOL = 1.0
    mag = -2.5*np.log10(f)
    sigma_m = 2.5*LOG_CONST*df['flux_error'].values/f
    xx = pd.DataFrame({'mjd':df['mjd'], 'm':mag, 's':sigma_m, 'filter':df['filter']})
    g = xx[xx['filter']=='ztfg'][['mjd','m','s']].rename(columns={'m':'m_g','s':'s_g'})
    r = xx[xx['filter']=='ztfr'][['mjd','m','s']].rename(columns={'m':'m_r','s':'s_r'})
    i = xx[xx['filter']=='ztfi'][['mjd','m','s']].rename(columns={'m':'m_i','s':'s_i'})
    g2r = pd.merge_asof(g.sort_values('mjd'), r.sort_values('mjd'),
                        on='mjd', direction='nearest', tolerance=TOL)
    r2i = pd.merge_asof(r.sort_values('mjd'), i.sort_values('mjd'),
                        on='mjd', direction='nearest', tolerance=TOL)
    out[['g_r','g_r_err']] = np.nan
    out[['r_i','r_i_err']] = np.nan
    idx_g = out['band_id']==BAND2ID['ztfg']
    out.loc[idx_g, ['g_r','g_r_err']] = np.column_stack([
        g2r['m_g'] - g2r['m_r'], np.sqrt(g2r['s_g']**2 + g2r['s_r']**2)
    ])
    idx_r = out['band_id']==BAND2ID['ztfr']
    out.loc[idx_r, ['r_i','r_i_err']] = np.column_stack([
        r2i['m_r'] - r2i['m_i'], np.sqrt(r2i['s_r']**2 + r2i['s_i']**2)
    ])
    out['has_g_r'] = out['g_r'].notna().astype(np.float32)
    out['has_r_i'] = out['r_i'].notna().astype(np.float32)
    return out

# ----------------------------
# Global/context features
# ----------------------------
CTX_GLOBAL_KEYS = [
    'days_since_peak','days_to_peak','peakmag_so_far','maxmag_so_far'
]

def context_metrics_up_to(merged_all: pd.DataFrame, jd_cut: float) -> Dict[str, float]:
    df = merged_all[merged_all['jd'] <= jd_cut]
    out = {k: np.nan for k in CTX_GLOBAL_KEYS}
    if len(df) == 0:
        return out
    peak_idx = df['flux'].idxmax()
    peak_row = df.loc[peak_idx]
    first_jd = df['jd'].min()
    last_jd  = df['jd'].max()
    mag = -2.5*np.log10(np.clip(df['flux'].values, 1e-12, None))
    out['days_since_peak'] = float(last_jd - peak_row['jd']) if last_jd != peak_row['jd'] else 0.0
    out['days_to_peak']    = float(peak_row['jd'] - first_jd)
    out['peakmag_so_far']  = float(np.nanmin(mag))
    out['maxmag_so_far']   = float(np.nanmax(mag))
    return out

def counts_per_filter_up_to(merged_all: pd.DataFrame, jd_cut: float) -> Dict[str, int]:
    df = merged_all[(merged_all['jd'] <= jd_cut) & (merged_all['fid'].isin(FIDS))]
    out = {'n_photometry_total': int(len(df))}
    for fid in FIDS:
        out[f'n_photometry_fid_{fid}'] = int(np.sum(df['fid']==fid))
    return out

# ----------------------------
# Alert stuff: Search for alert matching a point in the light curve
# ----------------------------
class AlertIndex:
    """Nearest/Best-by-time index over alerts per filter with robust cutout loader + extra selectors."""
    def __init__(self, alerts: List[dict], require_all_3: bool = True):
        self.require_all_3 = require_all_3
        good = []
        for a in alerts:
            try:
                c = a.get('candidate', a)
                jd = float(c['jd'])
                fid = int(c['fid'])
                if fid not in FIDS:
                    continue
                _ = a['cutoutScience']['stampData']
                _ = a['cutoutTemplate']['stampData']
                _ = a['cutoutDifference']['stampData']
                b = dict(a)
                b['_jd'] = jd
                b['_fid'] = fid
                try:
                    b['_sigmapsf'] = float(c.get('sigmapsf', np.inf))
                except Exception:
                    b['_sigmapsf'] = np.inf
                good.append(b)
            except Exception:
                continue
        self.by_fid: Dict[int, Dict[str, list]] = {}
        for fid in FIDS:
            aa = [a for a in good if int(a['_fid']) == fid]
            aa.sort(key=lambda x: x['_jd'])
            self.by_fid[fid] = {
                'jd': np.array([float(a['_jd']) for a in aa], dtype=np.float64),
                'alerts': aa
            }

    @staticmethod
    def _to_image(stamp: Union[bytes, bytearray, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(stamp, np.ndarray):
            return stamp.astype(np.float32)
        if isinstance(stamp, (bytes, bytearray)):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(stamp)) as gz:
                    raw = gz.read()
                with fits.open(io.BytesIO(raw)) as hdul:
                    return hdul[0].data.astype(np.float32)
            except Exception:
                pass
            try:
                with fits.open(io.BytesIO(stamp)) as hdul:
                    return hdul[0].data.astype(np.float32)
            except Exception:
                pass
            try:
                arr = np.load(io.BytesIO(stamp), allow_pickle=True)
                if isinstance(arr, np.ndarray):
                    return arr.astype(np.float32)
            except Exception:
                pass
        return None

    def _pack_to_triplet(self, a: dict) -> Optional[Tuple[np.ndarray, dict, float]]:
        try:
            sci  = self._to_image(a['cutoutScience']['stampData'])
            tmpl = self._to_image(a['cutoutTemplate']['stampData'])
            diff = self._to_image(a['cutoutDifference']['stampData'])
            if self.require_all_3 and (sci is None or tmpl is None or diff is None):
                return None
            if sci is None or tmpl is None or diff is None:
                return None
            img = np.stack([sci, tmpl, diff], axis=0)
            meta = dict(a.get('candidate', a))
            return img, meta, float(a['_jd'])
        except Exception:
            return None

    def get_best_in_window_by_sig(self, fid: int, jd: float, tol_days: float
                                  ) -> Optional[Tuple[np.ndarray, dict, float, float]]:
        pack = self.by_fid.get(fid, None)
        if pack is None or len(pack['jd']) == 0:
            return None
        jds = pack['jd']
        lo = np.searchsorted(jds, jd - tol_days, side='left')
        hi = np.searchsorted(jds, jd + tol_days, side='right')
        if hi <= lo:
            return None
        window = pack['alerts'][lo:hi]
        window.sort(key=lambda a: float(a.get('_sigmapsf', np.inf)))
        for a in window:
            trip = self._pack_to_triplet(a)
            if trip is None:
                continue
            img, meta, ajd = trip
            dt = abs(ajd - jd)
            return img, meta, ajd, dt
        return None

    def get_nearest_any(self, fid: int, jd: float) -> Optional[Tuple[np.ndarray, dict, float, float]]:
        pack = self.by_fid.get(fid, None)
        if pack is None or len(pack['jd']) == 0:
            return None
        jds = pack['jd']
        idx = np.searchsorted(jds, jd)
        cand_idxs = []
        if idx < len(jds): cand_idxs.append(idx)
        if idx-1 >= 0:     cand_idxs.append(idx-1)
        best = None; best_dt = 1e9; best_jd = None
        for k in cand_idxs:
            a = pack['alerts'][k]
            trip = self._pack_to_triplet(a)
            if trip is None:
                continue
            _, _, ajd = trip
            dt = abs(ajd - jd)
            if dt < best_dt:
                best_dt = dt; best = trip; best_jd = ajd
        if best is None:
            return None
        img, meta, ajd = best
        return img, meta, float(ajd), float(abs(ajd - jd))

# ----------------------------
# Spectra stuff: I/O & preprocessing
# ----------------------------
def _read_spectra_df(obj_id: str, data_dir: Path) -> pd.DataFrame:
    p = data_dir/obj_id/'spectra.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if 'ZTFID' in df.columns:
            df = df[(df['ZTFID'].astype(str) == str(obj_id)) | (~df['ZTFID'].notna())]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _extract_spectrum_time_mjd(sdf: pd.DataFrame) -> Optional[float]:
    if len(sdf) == 0:
        return None
    mjd_cols = ['observed_at_mjd', 'mjd', 'MJD', 'MJD_OBS', 'mjd_obs', 'spec_mjd', 'MJD-OBS', 'mjd-obs']
    jd_cols  = ['jd', 'JD', 'obs_jd', 'JD_OBS']
    for c in mjd_cols:
        if c in sdf.columns:
            try:
                v = pd.to_numeric(sdf[c], errors='coerce')
                if np.isfinite(v).any():
                    return float(np.nanmedian(v))
            except Exception:
                pass
    for c in jd_cols:
        if c in sdf.columns:
            try:
                v = pd.to_numeric(sdf[c], errors='coerce')
                if np.isfinite(v).any():
                    return float(np.nanmedian(v) - 2400000.5)
            except Exception:
                pass
    if 'observed_at' in sdf.columns:
        try:
            iso_vals = sdf['observed_at'].dropna().astype(str)
            if len(iso_vals):
                t = Time(iso_vals.iloc[0], format='isot', scale='utc')
                return float(t.mjd)
        except Exception:
            pass
    return None

def preprocess_spectra_df(sdf: pd.DataFrame, wave_grid: np.ndarray) -> Optional[np.ndarray]:
    if len(sdf) == 0:
        return None
    if not {'wavelength','flux'}.issubset(set(sdf.columns)):
        wl_col = None; fx_col = None
        for cand in ['wavelength','wave','lambda','lam','wl','Wavelength']:
            if cand in sdf.columns: wl_col = cand; break
        for cand in ['flux','Flux','FLUX','fluxcal','flam']:
            if cand in sdf.columns: fx_col = cand; break
        if wl_col is None or fx_col is None:
            return None
        sdf = sdf.rename(columns={wl_col:'wavelength', fx_col:'flux'})
    df = sdf[['wavelength','flux']].copy()
    df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
    df['flux']       = pd.to_numeric(df['flux'], errors='coerce')
    df = df.dropna(subset=['wavelength','flux'])
    if len(df) < 2:
        return None
    df = df.sort_values('wavelength')
    x = df['wavelength'].to_numpy(dtype=np.float64)
    y = df['flux'].to_numpy(dtype=np.float64)
    y_grid = _interp_with_extrap(x, y, wave_grid.astype(np.float64))
    mean = float(np.nanmean(y_grid))
    mad  = _mad(y_grid)
    if not np.isfinite(mad) or mad == 0.0:
        std = float(np.nanstd(y_grid))
        scale = std if (np.isfinite(std) and std > 0) else 1.0
    else:
        scale = mad
    y_grid = (y_grid - mean) / scale
    return y_grid.astype(np.float32)

# ----------------------------
# Build one object -> NPZ
# ----------------------------
ALERT_META_KEEP = [
    'sgscore1','sgscore2','distpsnr1','distpsnr2','nmtchps','sharpnr','scorr',
    'ra','dec','diffmaglim','sky','ndethist','ncovhist','sigmapsf','chinr',
    'magpsf','classtar','fid','rb','chipsf','distnr','magnr','ranr','decnr',
    'fwhm','srmag1','sgmag1','simag1','szmag1','srmag2','sgmag2','simag2','szmag2',
    'clrcoeff','clrcounc','zpclrcov'
]

def build_multimodal_for_object(obj_id: str,
                                label_int: int,
                                label_str: Optional[str],
                                out_dir: Path,
                                cfg: Config) -> Optional[Dict]:
    rng = np.random.RandomState(cfg.random_seed)  # not strictly needed, but keeps parity

    # PHOTOMETRY
    photo = load_photometry_obj(obj_id, cfg.data_dir)
    if len(photo) == 0:
        return None
    merged = merge_by_filter(photo, cfg.delta_t_hours)
    if len(merged) == 0:
        return None
    events = build_event_features(merged)
    if len(events) == 0:
        return None

    # ALERTS
    alerts_path = cfg.data_dir/obj_id/'alerts.npy'
    if not alerts_path.exists():
        return None
    alerts = np.load(alerts_path, allow_pickle=True)
    alerts = list(alerts) if isinstance(alerts, np.ndarray) else alerts
    idx = AlertIndex(alerts, require_all_3=cfg.require_all_3_cuts)

    images, meta_rows, event_rows, jds, fids, prov_rows = [], [], [], [], [], []
    keep_cols = [c for c in events.columns if c not in ('obj_id','jd','fid')]

    last_choice = {fid: None for fid in FIDS}  # carry-forward by filter

    for _, row in events.iterrows():
        fid = int(row['fid'])
        jd  = float(row['jd'])

        # first, pick best by smallest sigmapsf within \pm tol
        pick = idx.get_best_in_window_by_sig(fid, jd, cfg.alert_tol_days)
        policy = 'in_window_min_sigmapsf'

        # carry-forward last cutouts in that filter if nothing in window
        if pick is None and last_choice.get(fid) is not None:
            last = last_choice[fid]
            img, ameta, ajd = last['img'], last['meta'], last['jd_alert']
            dt_days = abs(jd - ajd)
            policy = 'fallback_last_in_filter'
        else:
            if pick is None and cfg.allow_fallback_nearest_any:
                near = idx.get_nearest_any(fid, jd)
                if near is not None:
                    img, ameta, ajd, dt_days = near
                    if (cfg.max_nearest_any_dt_days is not None) and (abs(dt_days) > cfg.max_nearest_any_dt_days):
                        near = None
                if near is not None:
                    policy = 'fallback_nearest_any'
                    pick = near

            if pick is None:
                continue  # cannot attach anything
            img, ameta, ajd, dt_days = pick

        # Build metadata vectors
        ctx_g = context_metrics_up_to(merged, jd)
        ctx_c = counts_per_filter_up_to(merged, jd)
        meta_vals = []
        for k in ALERT_META_KEEP:
            v = ameta.get(k, -999.0)
            if k == 'ra':  v = float(v)/180.0 - 1.0
            if k == 'dec': v = float(v)/90.0
            try:
                meta_vals.append(float(v))
            except Exception:
                meta_vals.append(-999.0)

        extra = {
            'days_since_peak': ctx_g['days_since_peak'],
            'days_to_peak': ctx_g['days_to_peak'],
            'age_sum_days': (ctx_g['days_since_peak'] + ctx_g['days_to_peak']),
            'peakmag_so_far': ctx_g['peakmag_so_far'],
            'maxmag_so_far': ctx_g['maxmag_so_far'],
            'max_over_peak_mag': (ctx_g['maxmag_so_far']/ctx_g['peakmag_so_far'])
                                 if (not np.isnan(ctx_g['peakmag_so_far']) and ctx_g['peakmag_so_far'] != 0) else np.nan,
            **ctx_c
        }
        extra_keys = list(extra.keys())
        extra_vals = [(-999.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
                      for v in extra.values()]

        images.append(img.astype(np.float32))
        meta_rows.append(np.array(meta_vals + extra_vals, dtype=np.float32))
        event_rows.append(row[keep_cols].to_numpy(dtype=np.float32))
        jds.append(jd); fids.append(fid)
        prov_rows.append({
            'jd_event': jd,
            'fid': fid,
            'jd_alert': float(ajd),
            'alert_dt_days': float(dt_days),           # alert.npy times since first detection
            'alert_matched': 1 if policy == 'in_window_min_sigmapsf' else 0,
            'select_policy': policy
        })

        last_choice[fid] = {'img': img, 'meta': ameta, 'jd_alert': float(ajd)}

    if len(images) == 0:
        return None

    # SPECTRA
    spec_df   = _read_spectra_df(obj_id, cfg.data_dir)
    spec_flux = preprocess_spectra_df(spec_df, cfg.wave_grid())
    spec_mjd_abs = _extract_spectrum_time_mjd(spec_df)

    photo_mjd0_abs = float(photo['jd'].min() - 2400000.5)  # first photo observation in absolute MJD
    spec_dt = float(spec_mjd_abs - photo_mjd0_abs) if spec_mjd_abs is not None else np.nan
    spec_jd = float(spec_mjd_abs + 2400000.5) if spec_mjd_abs is not None else np.nan

    if spec_flux is None:
        spectrum_vec  = np.zeros((0,), dtype=np.float32)
        spectrum_wave = np.zeros((0,), dtype=np.float32)
        has_spectrum  = np.int8(0)
    else:
        spectrum_vec  = spec_flux.astype(np.float32)
        spectrum_wave = cfg.wave_grid().astype(np.float32)
        has_spectrum  = np.int8(1)

    # Order by event time
    order = np.argsort(np.asarray(jds))
    images_arr = np.stack(images, axis=0)[order]
    event_arr  = np.vstack(event_rows).astype(np.float32)[order]
    meta_arr   = np.vstack(meta_rows).astype(np.float32)[order]
    jds_arr    = np.asarray(jds, dtype=np.float64)[order]
    fids_arr   = np.asarray(fids, dtype=np.int16)[order]
    prov_arr   = np.asarray(prov_rows, dtype=object)[order]

    event_cols = keep_cols
    meta_cols  = ALERT_META_KEEP + extra_keys

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/f"{obj_id}.npz"
    np.savez_compressed(
        out_path,
        images=images_arr,
        event_data=event_arr,
        event_columns=np.array(event_cols, dtype='U'),
        meta_data=meta_arr,
        meta_columns=np.array(meta_cols, dtype='U'),
        jd=jds_arr,
        fid=fids_arr,
        label=np.int64(label_int),
        label_str=np.array(label_str if label_str is not None else "", dtype='U'),
        provenance=prov_arr,
        spectrum=spectrum_vec,
        spectrum_wavelength=spectrum_wave,
        spectrum_dt=np.array(spec_dt, dtype=np.float64),
        spectrum_jd=np.array(spec_jd, dtype=np.float64),
        has_spectrum=np.array(has_spectrum, dtype=np.int8)
    )
    return {
        'object_id': obj_id,
        'filepath': str(out_path),
        'label': int(label_int),
        'label_str': label_str if label_str is not None else "",
        'n_events': int(images_arr.shape[0])
    }

# ----------------------------
# Manifests & stats
# ----------------------------
def safe_manifest(rows):
    cols = ['object_id','filepath','label','label_str','n_events']
    if len(rows)==0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    keep = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    return df[keep]

def write_manifest_csv(rows, path: Path, name=''):
    df = safe_manifest(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {name or path.name} -> {path}  (rows={len(df)})")
    return df

def compute_feature_stats_safe(manifest_csv: Path, kind: str, out_dir: Path):
    if not manifest_csv.exists() or os.path.getsize(manifest_csv) == 0:
        print(f"[stats:{kind}] skip -> missing or empty: {manifest_csv}")
        return
    try:
        man = pd.read_csv(manifest_csv)
    except pd.errors.EmptyDataError:
        print(f"[stats:{kind}] skip -> empty manifest: {manifest_csv}")
        return
    if 'filepath' not in man.columns or len(man)==0:
        print(f"[stats:{kind}] skip -> no rows.")
        return

    sum_=None; sumsq_=None; total=0; cols=None
    for path in tqdm(man['filepath'], desc=f'stats:{kind}'):
        if not Path(path).exists():
            continue
        npz = np.load(path, allow_pickle=True)
        data = npz['event_data'] if kind=='event' else npz['meta_data']
        if data.size==0:
            continue
        if cols is None:
            cols = npz['event_columns'] if kind=='event' else npz['meta_columns']
        if sum_ is None:
            sum_ = data.sum(axis=0); sumsq_=(data**2).sum(axis=0)
        else:
            sum_ += data.sum(axis=0); sumsq_ += (data**2).sum(axis=0)
        total += data.shape[0]
    if total == 0:
        print(f"[stats:{kind}] skip -> no data rows across files.")
        return
    mean = sum_/total
    var  = sumsq_/total - mean**2
    std  = np.sqrt(np.clip(var, 0, None))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir/f'feature_stats_{kind}.npz',
             columns=np.array(cols), mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"[stats:{kind}] wrote {out_dir/f'feature_stats_{kind}.npz'}")

def find_available_ids(spec_csv: Path, data_dir: Path, id_cols=("object_id","obj_id")) -> pd.DataFrame:
    spec = pd.read_csv(spec_csv)
    id_col = None
    for c in id_cols:
        if c in spec.columns:
            id_col = c; break
    if id_col is None:
        raise ValueError(f"Could not find ID column among {id_cols}; spec has {list(spec.columns)}.")
    spec = spec.rename(columns={id_col: 'object_id'})
    have = []
    for oid in tqdm(spec['object_id'].unique(), desc="Checking local files"):
        p = data_dir/oid/"photometry.csv"
        a = data_dir/oid/"alerts.npy"
        if p.exists() and a.exists():
            have.append(oid)
    out = spec[spec['object_id'].isin(have)].copy()
    return out

# ----------------------------
# Build all convenience
# ----------------------------
def build_all_preprocessed(cfg: Config) -> pd.DataFrame:
    out_root = Path(cfg.output_root)
    out_all  = out_root/'all'
    out_all.mkdir(parents=True, exist_ok=True)

    spec_avail = find_available_ids(cfg.spec_csv, cfg.data_dir)
    print(f"\nAvailable locally: {spec_avail['object_id'].nunique()} objects, "
          f"{spec_avail['type'].nunique()} classes.")

    classes  = sorted(spec_avail['type'].unique().tolist())
    label2id = {c:i for i,c in enumerate(classes)}
    spec_avail['label_int'] = spec_avail['type'].map(label2id)

    recs = []
    for oid, typ, lab in tqdm(spec_avail[['object_id','type','label_int']].itertuples(index=False),
                              total=len(spec_avail), desc="Building multimodal"):
        try:
            r = build_multimodal_for_object(oid, int(lab), typ, out_all, cfg)
            if r is not None and r.get('n_events', 0) > 0:
                row = {
                    'object_id': r.get('obj_id', oid),
                    'filepath' : r.get('filepath', str(out_all/f"{oid}.npz")),
                    'label'    : int(r.get('label', int(lab))),
                    'label_str': typ,
                    'n_events' : int(r.get('n_events', 0))
                }
                try:
                    z = np.load(row['filepath'], allow_pickle=True)
                    row['has_spectrum'] = int(np.array(z.get('has_spectrum', 0)).item())
                    if 'spectrum_dt' in z.files:
                        row['spectrum_dt'] = float(np.array(z['spectrum_dt']).item())
                except Exception:
                    row['has_spectrum'] = 0
                recs.append(row)
        except Exception as e:
            print(f"{oid} failed: {e}")

    built = write_manifest_csv(recs, out_root/'built_all.csv', name='built_all.csv')
    print(f"\nBuilt objects: {len(built)}")
    return built

__all__ = [
    'Config',
    'build_all_preprocessed',
    'build_multimodal_for_object',
    'compute_feature_stats_safe',
    'find_available_ids',
]


# =========================
# SPLITTING
# =========================
from sklearn.model_selection import train_test_split

def make_splits_from_manifest(
    built_csv: Path,
    out_root: Path,
    *,
    min_per_class: int = 7,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int,
    strict_stratify: bool = True
):
    """
    Read built_all.csv, filter to classes with >= min_per_class examples,
    then make (train/val/test) splits. If strict_stratify=True and
    stratification is not feasible, raise; otherwise fall back to seeded random.

    Writes:
      - splits.csv (object_id, split, label_str)
      - manifest_{train,val,test}.csv (no file copying; paths point to /all/*.npz)
      - feature_stats_{event,meta}.npz on train (if train not empty).
    """
    out_root = Path(out_root)
    built = pd.read_csv(built_csv)
    if len(built)==0:
        raise RuntimeError(f"No rows in {built_csv}; build first.")

    # ensure we have label_str
    if 'label_str' not in built.columns:
        raise RuntimeError("Manifest must contain 'label_str'. Rebuild or merge labels before splitting.")

    # filter by representation
    counts = built['label_str'].value_counts().sort_values()
    keep_classes = counts[counts >= min_per_class].index.tolist()
    filtered = built[built['label_str'].isin(keep_classes)].copy()
    dropped  = built[~built['label_str'].isin(keep_classes)]
    print(f"Keeping {len(keep_classes)} classes with ≥{min_per_class} examples "
          f"-> {filtered['object_id'].nunique()} objects. Dropped {dropped['object_id'].nunique()} objects.")

    if len(filtered)==0:
        raise RuntimeError("After filtering by min_per_class, no data remains.")

    # try stratified split
    ids  = filtered['object_id'].values
    labs = filtered['label_str'].values
    can_strat = (filtered['label_str'].value_counts().min() >= 2) and (filtered['label_str'].nunique() >= 2)

    if can_strat:
        tr, temp = train_test_split(ids, train_size=train_frac,
                                    stratify=labs, random_state=seed)
        labs_temp = filtered.set_index('object_id').loc[temp, 'label_str']
        # keep specified val/test ratio (normalize within the remaining fraction)
        rest = 1.0 - train_frac
        if rest <= 0:
            raise ValueError("train_frac must be < 1.0")
        val_share = val_frac / rest
        te_share  = test_frac / rest
        # guard against rounding edge cases
        if not np.isclose(val_share+te_share, 1.0):
            val_share = 0.5; te_share = 0.5
        va, te = train_test_split(temp, train_size=val_share,
                                  stratify=labs_temp, random_state=seed)
    else:
        if strict_stratify:
            raise ValueError("Stratified split not feasible with current min_per_class; "
                             "lower the threshold or set strict_stratify=False.")
        rng = np.random.RandomState(seed)
        rng.shuffle(ids)
        n_tr = int(round(train_frac * len(ids)))
        tr   = ids[:n_tr]
        temp = ids[n_tr:]
        rest = 1.0 - train_frac
        n_va = int(round((val_frac / rest) * len(temp))) if rest > 0 else 0
        va   = temp[:n_va]
        te   = temp[n_va:]

    # assemble splits dataframe
    rows = []
    rows += [(oid, 'train') for oid in tr]
    rows += [(oid, 'val')   for oid in va]
    rows += [(oid, 'test')  for oid in te]
    splits_df = pd.DataFrame(rows, columns=['object_id','split'])
    splits_df = splits_df.merge(filtered[['object_id','label_str']], on='object_id', how='left')

    splits_path = out_root/'splits.csv'
    splits_df.to_csv(splits_path, index=False)
    print(f"Wrote splits -> {splits_path}")

    # write per-split manifests (no file copying, keep original /all paths)
    manifests = {'train':[], 'val':[], 'test':[]}
    filt_idx = filtered.set_index('object_id')
    for split in ['train','val','test']:
        ids_s = splits_df.loc[splits_df['split']==split, 'object_id'].values
        rows_ = []
        for oid in ids_s:
            if oid not in filt_idx.index:
                continue
            r = filt_idx.loc[oid]
            rows_.append({
                'object_id': oid,
                'filepath' : r['filepath'],
                'label'    : int(r['label']) if 'label' in r else np.nan,
                'label_str': r['label_str'],
                'n_events' : int(r['n_events']) if 'n_events' in r else np.nan
            })
        write_manifest_csv(rows_, out_root/f'manifest_{split}.csv', name=f'manifest_{split}.csv')

    # compute train stats (if non-empty)
    train_manifest = out_root/'manifest_train.csv'
    if train_manifest.exists() and os.path.getsize(train_manifest)>0:
        compute_feature_stats_safe(train_manifest, 'event', out_root)
        compute_feature_stats_safe(train_manifest, 'meta',  out_root)
    print("Splitting complete.")
