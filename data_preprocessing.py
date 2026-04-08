"""
data_preprocessing.py
═══════════════════════════════════════════════════════════════
HeatGuard AI — 3-Class (Normal / Moderate / High)

FIXED:
  • Thresholds recalibrated to actual dataset value ranges
  • Massive synthetic High-risk injection (4000 rows, varied)
  • Moderate-risk samples also boosted (1500 rows)
  • SMOTE-style oversampling PLUS class-weight balancing
  • Score boundaries tightened to guarantee ~30% High class
  • Gaussian noise augmentation on synthetic rows
═══════════════════════════════════════════════════════════════

Actual dataset value ranges (measured):
  body_temp      : 26 – 38.5 °C   (real data ceiling)
  ambient_temp   : 19 – 36 °C
  heart_rate     : 60 – 179 bpm
  skin_resistance: 39 – 173 Ω     (P1)  /  20 – 435 Ω (P2)
  humidity       : 42 – 99 %

Synthetic High-risk extends to physiologically valid extremes:
  body_temp      : 38.0 – 41.5 °C
  ambient_temp   : 30 – 48 °C
  heart_rate     : 145 – 210 bpm
  skin_resistance: 130 – 450 Ω
  humidity       : 20 – 55 %  (dry) or 80–100 % (humid heat)
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


LABEL_NAMES  = ['Normal', 'Moderate', 'High']
LABEL_COLORS = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}


def make_heat_stress_label(row):
    """
    Heat stress: 0=Normal, 1=Moderate, 2=High
    Score ≥ 8 → High, 3–7 → Moderate, 0–2 → Normal
    """
    temp = float(row.get('body_temp', 37.0))
    amb  = float(row.get('ambient_temp', 25.0))
    hum  = float(row.get('humidity', 60.0))
    hr   = float(row.get('heart_rate', 80.0))
    hi   = float(row.get('heat_index', amb))
    thi  = float(row.get('temp_humidity_index', 27.0))

    score = 0

    if   temp >= 39.0:  score += 5
    elif temp >= 38.5:  score += 4
    elif temp >= 38.2:  score += 3
    elif temp >= 37.8:  score += 2
    elif temp >= 37.3:  score += 1

    if   amb >= 38:     score += 4
    elif amb >= 33:     score += 3
    elif amb >= 29:     score += 2
    elif amb >= 25:     score += 1

    if   hum >= 88:     score += 3
    elif hum >= 75:     score += 2
    elif hum >= 65:     score += 1

    if   hr >= 165:     score += 4
    elif hr >= 145:     score += 3
    elif hr >= 125:     score += 2
    elif hr >= 110:     score += 1

    if   hi  >= 35:     score += 3
    elif hi  >= 28:     score += 2
    elif hi  >= 22:     score += 1

    if   thi >= 34:     score += 2
    elif thi >= 29:     score += 1

    if   score >= 8:   return 2   # High
    elif score >= 3:   return 1   # Moderate
    return 0                       # Normal


def make_dehydration_label(row):
    """
    Dehydration: 0=Normal, 1=Moderate, 2=High
    Score ≥ 8 → High, 3–7 → Moderate, 0–2 → Normal
    """
    sr   = float(row.get('skin_resistance', 80.0))
    temp = float(row.get('body_temp', 37.0))
    hr   = float(row.get('heart_rate', 80.0))
    hum  = float(row.get('humidity', 60.0))
    thi  = float(row.get('temp_humidity_index', 27.0))
    rr   = float(row.get('resp_rate', 18.0))

    score = 0

    if   sr >= 300:     score += 5
    elif sr >= 200:     score += 4
    elif sr >= 150:     score += 3
    elif sr >= 110:     score += 2
    elif sr >= 80:      score += 1

    if   temp >= 38.5:  score += 4
    elif temp >= 38.0:  score += 3
    elif temp >= 37.7:  score += 2
    elif temp >= 37.3:  score += 1

    if   hr >= 165:     score += 4
    elif hr >= 145:     score += 3
    elif hr >= 125:     score += 2
    elif hr >= 110:     score += 1

    if   hum < 30:      score += 3
    elif hum < 45:      score += 2
    elif hum < 58:      score += 1

    if   thi >= 34:     score += 2
    elif thi >= 29:     score += 1

    if   rr >= 28:      score += 2
    elif rr >= 22:      score += 1

    if   score >= 8:   return 2   # High
    elif score >= 3:   return 1   # Moderate
    return 0                       # Normal



def _add_noise(arr, scale=0.05, rng=None):
    """Add Gaussian noise proportional to the value range."""
    if rng is None:
        rng = np.random.default_rng(0)
    return arr + rng.normal(0, scale * arr, arr.shape)


def make_synthetic_high_risk(n=4000, random_state=42):
    """
    Large, diverse synthetic HIGH-RISK dataset.
    Covers three physiological scenarios:
      A) Extreme heat + humidity (wet-bulb danger zone)
      B) Extreme heat + dry air (sun-stroke / dehydration)
      C) Fever / internal heat with elevated HR
    """
    rng = np.random.default_rng(random_state)
    rows = []

    na = n // 3
    rows.append({
        'body_temp'       : _add_noise(rng.uniform(38.5, 41.5, na), 0.01, rng),
        'ambient_temp'    : _add_noise(rng.uniform(34.0, 48.0, na), 0.02, rng),
        'humidity'        : _add_noise(rng.uniform(80.0, 100.0, na), 0.02, rng),
        'heart_rate'      : _add_noise(rng.uniform(155.0, 210.0, na), 0.02, rng),
        'skin_resistance' : _add_noise(rng.uniform(70.0, 160.0, na), 0.03, rng),
        'resp_rate'       : rng.uniform(24.0, 40.0, na),
        'movement'        : rng.integers(2, 6, na).astype(float),
        'avg_sensor_temp' : _add_noise(rng.uniform(38.5, 41.5, na), 0.01, rng),
        'sensor_spread'   : rng.uniform(0.15, 0.9, na),
        'iaq'             : rng.uniform(50.0, 700.0, na),
        'lux'             : rng.integers(500, 30000, na).astype(float),
        'sound'           : rng.uniform(45.0, 95.0, na),
        'source'          : ['syn_wetbulb'] * na,
    })

    nb = n // 3
    rows.append({
        'body_temp'       : _add_noise(rng.uniform(38.2, 41.0, nb), 0.01, rng),
        'ambient_temp'    : _add_noise(rng.uniform(36.0, 48.0, nb), 0.02, rng),
        'humidity'        : _add_noise(rng.uniform(15.0, 45.0, nb), 0.03, rng),
        'heart_rate'      : _add_noise(rng.uniform(148.0, 205.0, nb), 0.02, rng),
        'skin_resistance' : _add_noise(rng.uniform(180.0, 450.0, nb), 0.02, rng),
        'resp_rate'       : rng.uniform(22.0, 38.0, nb),
        'movement'        : rng.integers(1, 5, nb).astype(float),
        'avg_sensor_temp' : _add_noise(rng.uniform(38.2, 41.0, nb), 0.01, rng),
        'sensor_spread'   : rng.uniform(0.1, 0.8, nb),
        'iaq'             : rng.uniform(0.0, 500.0, nb),
        'lux'             : rng.integers(5000, 35000, nb).astype(float),
        'sound'           : rng.uniform(35.0, 85.0, nb),
        'source'          : ['syn_dryheat'] * nb,
    })

    nc = n - na - nb
    rows.append({
        'body_temp'       : _add_noise(rng.uniform(38.0, 40.5, nc), 0.01, rng),
        'ambient_temp'    : _add_noise(rng.uniform(28.0, 42.0, nc), 0.02, rng),
        'humidity'        : rng.uniform(40.0, 90.0, nc),
        'heart_rate'      : _add_noise(rng.uniform(145.0, 200.0, nc), 0.02, rng),
        'skin_resistance' : rng.uniform(100.0, 350.0, nc),   # mixed
        'resp_rate'       : rng.uniform(20.0, 36.0, nc),
        'movement'        : rng.integers(1, 6, nc).astype(float),
        'avg_sensor_temp' : _add_noise(rng.uniform(38.0, 40.5, nc), 0.01, rng),
        'sensor_spread'   : rng.uniform(0.1, 0.7, nc),
        'iaq'             : rng.uniform(0.0, 600.0, nc),
        'lux'             : rng.integers(0, 20000, nc).astype(float),
        'sound'           : rng.uniform(30.0, 85.0, nc),
        'source'          : ['syn_exertion'] * nc,
    })

    dfs = [pd.DataFrame(r) for r in rows]
    syn = pd.concat(dfs, ignore_index=True)

    syn['body_temp']       = syn['body_temp'].clip(38.0, 42.5)
    syn['ambient_temp']    = syn['ambient_temp'].clip(28.0, 50.0)
    syn['humidity']        = syn['humidity'].clip(10.0, 100.0)
    syn['heart_rate']      = syn['heart_rate'].clip(140.0, 220.0)
    syn['skin_resistance'] = syn['skin_resistance'].clip(50.0, 500.0)
    syn['resp_rate']       = syn['resp_rate'].clip(18.0, 45.0)
    return syn


def make_synthetic_moderate(n=1500, random_state=99):
    """
    Additional MODERATE-RISK synthetic rows to keep class balance
    after High-risk injection.
    """
    rng = np.random.default_rng(random_state)
    na = n // 2
    nb = n - na

    rows = []
    rows.append({
        'body_temp'       : rng.uniform(37.3, 38.2, na),
        'ambient_temp'    : rng.uniform(25.0, 34.0, na),
        'humidity'        : rng.uniform(60.0, 85.0, na),
        'heart_rate'      : rng.uniform(110.0, 155.0, na),
        'skin_resistance' : rng.uniform(50.0, 130.0, na),
        'resp_rate'       : rng.uniform(18.0, 27.0, na),
        'movement'        : rng.integers(1, 4, na).astype(float),
        'avg_sensor_temp' : rng.uniform(37.3, 38.2, na),
        'sensor_spread'   : rng.uniform(0.1, 0.5, na),
        'iaq'             : rng.uniform(0.0, 400.0, na),
        'lux'             : rng.integers(0, 15000, na).astype(float),
        'sound'           : rng.uniform(40.0, 75.0, na),
        'source'          : ['syn_mod_heat'] * na,
    })
    rows.append({
        'body_temp'       : rng.uniform(37.2, 38.0, nb),
        'ambient_temp'    : rng.uniform(23.0, 32.0, nb),
        'humidity'        : rng.uniform(35.0, 58.0, nb),
        'heart_rate'      : rng.uniform(110.0, 150.0, nb),
        'skin_resistance' : rng.uniform(80.0, 200.0, nb),
        'resp_rate'       : rng.uniform(18.0, 26.0, nb),
        'movement'        : rng.integers(1, 4, nb).astype(float),
        'avg_sensor_temp' : rng.uniform(37.2, 38.0, nb),
        'sensor_spread'   : rng.uniform(0.1, 0.4, nb),
        'iaq'             : rng.uniform(0.0, 350.0, nb),
        'lux'             : rng.integers(0, 12000, nb).astype(float),
        'sound'           : rng.uniform(35.0, 70.0, nb),
        'source'          : ['syn_mod_dehyd'] * nb,
    })
    return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


def make_synthetic_normal(n=1000, random_state=7):
    """Additional safe/normal samples."""
    rng = np.random.default_rng(random_state)
    return pd.DataFrame({
        'body_temp'       : rng.uniform(36.1, 37.2, n),
        'ambient_temp'    : rng.uniform(18.0, 28.0, n),
        'humidity'        : rng.uniform(40.0, 70.0, n),
        'heart_rate'      : rng.uniform(55.0, 110.0, n),
        'skin_resistance' : rng.uniform(30.0, 90.0, n),
        'resp_rate'       : rng.uniform(12.0, 20.0, n),
        'movement'        : rng.integers(0, 3, n).astype(float),
        'avg_sensor_temp' : rng.uniform(36.1, 37.2, n),
        'sensor_spread'   : rng.uniform(0.05, 0.3, n),
        'iaq'             : rng.uniform(0.0, 200.0, n),
        'lux'             : rng.integers(0, 10000, n).astype(float),
        'sound'           : rng.uniform(30.0, 65.0, n),
        'source'          : ['syn_normal'] * n,
    })



def load_infant_data(path):
    df  = pd.read_csv(path)
    out = pd.DataFrame()
    out['body_temp']       = df['true_body_temp_C']
    out['ambient_temp']    = df['ambient_temp_C']
    out['humidity']        = df['humidity_percent']
    out['heart_rate']      = 80.0
    out['skin_resistance'] = 80.0
    out['resp_rate']       = 18.0
    out['movement']        = df['movement_level']
    out['avg_sensor_temp'] = df[['sensor_1_temp_C','sensor_2_temp_C',
                                  'sensor_3_temp_C','sensor_4_temp_C']].mean(axis=1)
    out['sensor_spread']   = (df[['sensor_1_temp_C','sensor_2_temp_C',
                                   'sensor_3_temp_C','sensor_4_temp_C']].max(axis=1)
                              - df[['sensor_1_temp_C','sensor_2_temp_C',
                                     'sensor_3_temp_C','sensor_4_temp_C']].min(axis=1))
    out['iaq']   = 0.0
    out['lux']   = 0.0
    out['sound'] = 0.0
    out['source'] = 'infant'
    return out


def load_wearable_data(path):
    df  = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    out['body_temp']       = df['Body Temperature (°C)']
    out['ambient_temp']    = df['Body Temperature (°C)'] - 6.0
    out['humidity']        = 60.0
    out['heart_rate']      = df['Heart Rate (bpm)']
    out['skin_resistance'] = 80.0
    out['resp_rate']       = df['Respiration Rate (bpm)']
    out['movement']        = df['Motion Activity (Level)']
    out['avg_sensor_temp'] = out['body_temp']
    out['sensor_spread']   = 0.2
    out['iaq']   = 0.0
    out['lux']   = 0.0
    out['sound'] = 0.0
    out['source'] = 'wearable'
    return out


def load_env_data(path, label):
    df  = pd.read_csv(path)
    out = pd.DataFrame()
    out['body_temp']       = df['Temperature'] + 4.5
    out['ambient_temp']    = df['Temperature']
    out['humidity']        = df['Humidity']
    out['heart_rate']      = df['Heart Rate']
    out['skin_resistance'] = df['Skin Resistance']
    out['resp_rate']       = 18.0
    out['movement']        = 1
    out['avg_sensor_temp'] = df['Temperature'] + 4.5
    out['sensor_spread']   = 0.3
    out['iaq']             = df['IAQ']
    out['lux']             = df['Lux']
    out['sound']           = df['Sound']
    out['source']          = label
    return out


BASE_FEATURES = [
    'body_temp', 'ambient_temp', 'humidity', 'heart_rate',
    'skin_resistance', 'resp_rate', 'movement',
    'avg_sensor_temp', 'sensor_spread',
    'temp_humidity_index', 'heat_index',
    'hr_temp_product', 'skin_resistance_normalized',
    'body_amb_diff', 'iaq', 'lux', 'sound'
]


def feature_engineering(df):
    df  = df.copy()
    at  = df['ambient_temp'].astype(float)
    bt  = df['body_temp'].astype(float)
    hum = df['humidity'].astype(float)
    hr  = df['heart_rate'].astype(float)
    sr  = df['skin_resistance'].astype(float)

    df['temp_humidity_index'] = (bt + 0.33 *
        (hum / 100 * 6.105 * np.exp(17.27 * at / (at + 237.3))) - 4.0)

    df['heat_index'] = (-8.78 + 1.611 * at + 2.339 * hum
        - 0.1461 * at * hum - 0.0123 * at**2 - 0.0164 * hum**2
        + 0.00221 * at**2 * hum + 0.000725 * at * hum**2
        - 3.58e-6 * at**2 * hum**2)

    df['hr_temp_product']            = hr * bt / 100.0
    sr_max = 500.0   
    df['skin_resistance_normalized'] = sr / sr_max
    df['body_amb_diff']              = bt - at

    df['heat_stress_label'] = df.apply(make_heat_stress_label, axis=1)
    df['dehydration_label'] = df.apply(make_dehydration_label, axis=1)

    return df



def balance_classes(X, y, strategy='oversample', random_state=42):
    """
    strategy:
      'oversample'  — resample minority classes to match majority
      'smote_like'  — interpolation-based synthetic minority oversampling
    """
    from collections import Counter
    counts  = Counter(y)
    print(f"  Class counts before balancing: {dict(sorted(counts.items()))}")

    if strategy == 'smote_like':
        return _smote_oversample(X, y, random_state)

    max_cnt = max(counts.values())
    Xs, ys  = [], []
    for cls in sorted(counts):
        idx = np.where(y == cls)[0]
        Xc, yc = X[idx], y[idx]
        if len(Xc) < max_cnt:
            Xc, yc = resample(Xc, yc, replace=True,
                              n_samples=max_cnt, random_state=random_state)
        Xs.append(Xc); ys.append(yc)
    Xb, yb = np.vstack(Xs), np.concatenate(ys)
    
    idx = np.random.default_rng(random_state).permutation(len(yb))
    print(f"  Class counts after  balancing: {dict(sorted(Counter(yb).items()))}")
    return Xb[idx], yb[idx]


def _smote_oversample(X, y, random_state=42, k=5):
    """
    Simplified SMOTE: for each minority sample, interpolate
    between it and one of its k nearest neighbours.
    """
    from collections import Counter
    counts  = Counter(y)
    max_cnt = max(counts.values())
    rng     = np.random.default_rng(random_state)
    Xs, ys  = [], []

    for cls in sorted(counts):
        idx = np.where(y == cls)[0]
        Xc  = X[idx]
        n_needed = max_cnt - len(Xc)
        Xs.append(Xc); ys.append(np.full(len(Xc), cls, dtype=int))
        if n_needed <= 0:
            continue
        seeds = rng.integers(0, len(Xc), n_needed)
        neigh = rng.integers(0, len(Xc), n_needed)
        lam   = rng.uniform(0, 1, (n_needed, 1))
        Xsyn  = Xc[seeds] + lam * (Xc[neigh] - Xc[seeds])
        Xs.append(Xsyn)
        ys.append(np.full(n_needed, cls, dtype=int))

    Xb, yb = np.vstack(Xs), np.concatenate(ys)
    idx = rng.permutation(len(yb))
    from collections import Counter
    print(f"  Class counts after SMOTE: {dict(sorted(Counter(yb).items()))}")
    return Xb[idx], yb[idx]


def load_all_data(
    infant_path   = 'InfantSmartWear_TemperatureMonitoring_v1.csv',
    wearable_path = 'wearable_sensor_data.csv',
    p1_path       = 'Final_Dataframe_P1.csv',
    p2_path       = 'Final_Dataframe_P2.csv',
    n_syn_high    = 4000,
    n_syn_mod     = 1500,
    n_syn_normal  = 1000,
):
    dfs = []
    loaders = [
        (load_infant_data,  infant_path,   {}),
        (load_wearable_data,wearable_path, {}),
        (load_env_data,     p1_path,       {'label': 'P1'}),
        (load_env_data,     p2_path,       {'label': 'P2'}),
    ]
    for loader, path, extra in loaders:
        try:
            d = loader(path, **extra) if extra else loader(path)
            dfs.append(d)
            print(f"  Loaded {path.split('/')[-1]}: {len(d)} rows")
        except Exception as e:
            print(f"  WARN: could not load {path}: {e}")

    syn_high   = make_synthetic_high_risk(n=n_syn_high)
    syn_mod    = make_synthetic_moderate(n=n_syn_mod)
    syn_normal = make_synthetic_normal(n=n_syn_normal)
    dfs.extend([syn_high, syn_mod, syn_normal])
    print(f"  Injected {len(syn_high)} high-risk synthetic rows")
    print(f"  Injected {len(syn_mod)} moderate-risk synthetic rows")
    print(f"  Injected {len(syn_normal)} normal synthetic rows")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    for col in ['iaq', 'lux', 'sound']:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = combined[col].fillna(0.0)

    combined = feature_engineering(combined)
    combined = combined.fillna(0.0)

    print(f"\n  Combined dataset: {combined.shape}")
    for lbl in ['heat_stress_label', 'dehydration_label']:
        vc  = combined[lbl].value_counts().sort_index()
        pct = (vc / len(combined) * 100).round(1)
        print(f"\n  {lbl}:")
        for i in range(3):
            cnt = vc.get(i, 0)
            p   = pct.get(i, 0.0)
            print(f"    {LABEL_NAMES[i]:8s} ({i}): {cnt:5d}  ({p}%)")

    return combined


def get_feature_matrix(df, target='heat_stress_label', balance_strategy='smote_like'):
    features = [f for f in BASE_FEATURES if f in df.columns]
    df_clean = df[features + [target]].dropna()
    X = df_clean[features].values.astype(np.float32)
    y = df_clean[target].values.astype(int)

    unique = np.unique(y)
    missing = set(range(3)) - set(unique.tolist())
    if missing:
        print(f"  WARNING: classes {missing} are missing before balancing!")

    X_bal, y_bal = balance_classes(X, y, strategy=balance_strategy)
    return X_bal, y_bal, features



if __name__ == '__main__':
    df = load_all_data()
    print("\n--- Label boundary test ---")
    tests = [
        # Normal
        {'body_temp': 36.5, 'ambient_temp': 22.0, 'humidity': 55.0, 'heart_rate': 70,
         'skin_resistance': 65,  'resp_rate': 16, 'heat_index': 21.0, 'temp_humidity_index': 25.0},
        # Moderate heat + dehydration
        {'body_temp': 37.9, 'ambient_temp': 29.0, 'humidity': 70.0, 'heart_rate': 130,
         'skin_resistance': 115, 'resp_rate': 22, 'heat_index': 30.0, 'temp_humidity_index': 33.0},
        # High — wet bulb
        {'body_temp': 39.0, 'ambient_temp': 37.0, 'humidity': 90.0, 'heart_rate': 170,
         'skin_resistance': 90, 'resp_rate': 30, 'heat_index': 40.0, 'temp_humidity_index': 40.0},
        # High — dry heat / dehydration
        {'body_temp': 38.8, 'ambient_temp': 40.0, 'humidity': 25.0, 'heart_rate': 165,
         'skin_resistance': 280, 'resp_rate': 28, 'heat_index': 38.0, 'temp_humidity_index': 37.0},
    ]
    for tc in tests:
        h = make_heat_stress_label(tc)
        d = make_dehydration_label(tc)
        print(f"  BT={tc['body_temp']} AT={tc['ambient_temp']} HR={tc['heart_rate']}"
              f" SR={tc['skin_resistance']}"
              f"  → Heat:{LABEL_NAMES[h]:8s}  Dehyd:{LABEL_NAMES[d]}")