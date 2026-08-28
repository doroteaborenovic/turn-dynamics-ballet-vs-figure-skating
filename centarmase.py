import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter, butter, filtfilt

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PODEŠAVANJE DIREKTORIJUMA I BAZA ANTROPOMETRIJE
# =============================================================================

CANDIDATE_DIRS = [
    r"C:\Users\PC\gitara\projekat.fizika\skracene_koordinate",
    "skracene_koordinate",
    "kinematika_rezultati",
    "popravljene_koordinate",
    "konacne_koordinate",
    "."
]

INPUT_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(d) and len(glob.glob(os.path.join(d, "*.csv"))) > 0:
        INPUT_DIR = d
        break

if INPUT_DIR is None:
    INPUT_DIR = "skracene_koordinate"
    os.makedirs(INPUT_DIR, exist_ok=True)

DIR_PLOTS = "grafici_mos_dykstra_tacan"
DIR_TABLES = "rezultati_mos_dykstra_tacan"
os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)

FPS = 30.0
DT_DEFAULT = 1.0 / FPS

# Baza podataka o atletama
ATHLETE_DB = {
    "marianela":    {"height": 1.74, "weight": 52.0, "sport": "Balet", "shoe_size": 39},
    "kapitonova":   {"height": 1.68, "weight": 48.0, "sport": "Balet", "shoe_size": 38},
    "khoreva":      {"height": 1.73, "weight": 47.0, "sport": "Balet", "shoe_size": 39},
    "trusova":      {"height": 1.66, "weight": 50.0, "sport": "Umetničko klizanje", "shoe_size": 37},
    "valieva":      {"height": 1.60, "weight": 44.0, "sport": "Umetničko klizanje", "shoe_size": 36},
    "kamilavalieva":{"height": 1.60, "weight": 44.0, "sport": "Umetničko klizanje", "shoe_size": 36},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "sport": "Umetničko klizanje", "shoe_size": 36},
    "scerebakova":  {"height": 1.61, "weight": 42.0, "sport": "Umetničko klizanje", "shoe_size": 36},
    "liu":          {"height": 1.58, "weight": 45.0, "sport": "Umetničko klizanje", "shoe_size": 36}
}

# Dimenzije stopala prema broju obuće (u cm)
FOOT_DIMENSIONS_CM = {
    36: {"length": 22.9, "width": 8.4},
    37: {"length": 23.8, "width": 8.7},
    38: {"length": 24.3, "width": 8.9},
    39: {"length": 25.1, "width": 9.2},
    40: {"length": 25.4, "width": 9.4}
}

# De Leva 14-segmentni ženski model (1996)
DE_LEVA = {
    "Head":        {"mass": 0.0668, "pos": 0.5000},
    "Trunk":       {"mass": 0.4257, "pos": 0.4360},
    "R_UpperArm":  {"mass": 0.0255, "pos": 0.5754},
    "L_UpperArm":  {"mass": 0.0255, "pos": 0.5754},
    "R_Forearm":   {"mass": 0.0138, "pos": 0.4559},
    "L_Forearm":   {"mass": 0.0138, "pos": 0.4559},
    "R_Hand":      {"mass": 0.0056, "pos": 0.3400},
    "L_Hand":      {"mass": 0.0056, "pos": 0.3400},
    "R_Thigh":     {"mass": 0.1478, "pos": 0.3612},
    "L_Thigh":     {"mass": 0.1478, "pos": 0.3612},
    "R_Shank":     {"mass": 0.0481, "pos": 0.4416},
    "L_Shank":     {"mass": 0.0481, "pos": 0.4416},
    "R_Foot":      {"mass": 0.0129, "pos": 0.4014},
    "L_Foot":      {"mass": 0.0129, "pos": 0.4014}
}

def interp_seg(p_prox, p_dist, ratio):
    return p_prox + ratio * (p_dist - p_prox)

def filter_signal(arr, win=11):
    s = pd.Series(arr).interpolate(method='linear', limit_direction='both').ffill().bfill()
    clean = s.values
    w = min(win, len(clean) if len(clean) % 2 != 0 else len(clean) - 1)
    if w < 5: return clean
    return savgol_filter(clean, window_length=w, polyorder=2)

def clean_and_interpolate_signal(arr, win=9):
    arr = np.asarray(arr, dtype=float).copy()
    arr[arr == 0.0] = np.nan
    n = len(arr)
    frames = np.arange(n)
    
    diff = np.abs(np.diff(arr, prepend=arr[0]))
    threshold = np.nanmedian(diff) * 3.5
    if threshold > 0.03:
        arr[diff > threshold] = np.nan

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < 4:
        return pd.Series(arr).interpolate(method='linear', limit_direction='both').bfill().ffill().values
        
    pchip = PchipInterpolator(frames[valid_idx], arr[valid_idx], extrapolate=True)
    filled = pchip(frames)
    
    w = min(win, n if n % 2 != 0 else n - 1)
    if w >= 5:
        filled = savgol_filter(filled, window_length=w, polyorder=2)
    return filled

def fmt_signed(val):
    return f"{val:+.1f}"

# =============================================================================
# 2. GLAVNA OBRADA PO DYKSTRA ET AL. (2025) & HOF (2005)
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "fizika_" not in f and "Tabela" not in f]

print("\n" + "="*115)
print("DYKSTRA ET AL. (2025) / HOF (2005) STABILIZOVANA BIOMEHANIČKA EVALUACIJA")
print(f"Ulazni direktorijum: {os.path.abspath(INPUT_DIR)}")
print(f"Pronađeno CSV fajlova: {len(all_files)}")
print("="*115 + "\n")

summary_rows = []
all_data_for_comp = {}

for file_path in sorted(all_files):
    df_raw = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).lower()
    
    athlete_key = None
    for k in ATHLETE_DB:
        if k in file_name:
            athlete_key = k
            break
            
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "sport": "Klizanje/Balet", "shoe_size": 38})
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file_path).split('.')[0].upper()
    sport = athlete_data["sport"]
    height_m = athlete_data["height"]
    shoe_size = athlete_data["shoe_size"]

    is_wide_format = ("x_0" in df_raw.columns or "X_0" in df_raw.columns)
    if not is_wide_format and "Landmark_ID" not in df_raw.columns:
        continue

    n_frames = len(df_raw) if is_wide_format else len(df_raw['Frame'].unique())
    if n_frames < 15:
        continue
        
    time_raw = df_raw["timestamp_sec"].values if "timestamp_sec" in df_raw.columns else (
               df_raw["Time_s"].values if "Time_s" in df_raw.columns else np.arange(n_frames) * DT_DEFAULT)
    dt = np.median(np.diff(time_raw)) if len(time_raw) > 1 else DT_DEFAULT
    if dt <= 0: dt = DT_DEFAULT

    # 1. Rekonstrukcija 33 tačke
    if is_wide_format:
        pts_array = np.zeros((n_frames, 33, 3))
        for lm in range(33):
            for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
                col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_raw.columns else f"{ax_name.upper()}_{lm}"
                if col in df_raw.columns:
                    pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_raw[col].values)
    else:
        coord_cols = [c for c in ['X_clean', 'Y_clean', 'Z_clean', 'X_raw', 'Y_raw', 'Z_raw', 'X', 'Y', 'Z'] if c in df_raw.columns]
        xc, yc, zc = coord_cols[0], coord_cols[1], coord_cols[2]
        frames_list = sorted(df_raw['Frame'].unique())
        pts_array = np.zeros((len(frames_list), 33, 3))
        for f_i, f_idx in enumerate(frames_list):
            f_df = df_raw[df_raw['Frame'] == f_idx].set_index('Landmark_ID')
            for lm in range(33):
                if lm in f_df.index:
                    pts_array[f_i, lm, :] = f_df.loc[lm, [xc, yc, zc]].values

    # Kondicioniranje Z-ose
    span_x = np.ptp(pts_array[:, :, 0])
    span_z = np.ptp(pts_array[:, :, 2])
    if span_z > 2.0 * span_x and span_x > 0.05:
        pts_array[:, :, 2] *= (span_x / span_z)

    w_z = min(15, n_frames if n_frames % 2 != 0 else n_frames - 1)
    if w_z >= 5:
        for lm in range(33):
            pts_array[:, lm, 2] = savgol_filter(pts_array[:, lm, 2], window_length=w_z, polyorder=2)

    # 2. Detekcija stajne noge (Pivot)
    var_l = np.mean(np.var(pts_array[:, [27, 29, 31], :], axis=0))
    var_r = np.mean(np.var(pts_array[:, [28, 30, 32], :], axis=0))
    
    planted_side = "Leva" if var_l <= var_r else "Desna"
    p_ank = 27 if planted_side == "Leva" else 28
    p_toe = 31 if planted_side == "Leva" else 32

    # 3. Skaliranje na metričku visinu
    mid_shoulder = (pts_array[:, 11, :] + pts_array[:, 12, :]) / 2.0
    mid_hip = (pts_array[:, 23, :] + pts_array[:, 24, :]) / 2.0
    head_vertex = pts_array[:, 0, :] + 0.5 * (pts_array[:, 0, :] - mid_shoulder)
    knee_pt = pts_array[:, 25, :] if planted_side == "Leva" else pts_array[:, 26, :]
    hip_pt = pts_array[:, 23, :] if planted_side == "Leva" else pts_array[:, 24, :]
    ankle_pt = pts_array[:, p_ank, :]
    toe_pt = pts_array[:, p_toe, :]

    h_chain = (np.linalg.norm(head_vertex - mid_hip, axis=1) + 
               np.linalg.norm(hip_pt - knee_pt, axis=1) + 
               np.linalg.norm(knee_pt - ankle_pt, axis=1) +
               np.linalg.norm(ankle_pt - toe_pt, axis=1) * 0.35)
    
    scale = height_m / np.median(h_chain) if len(h_chain) > 0 and np.median(h_chain) > 0.1 else 1.0
    pts_m = pts_array * scale

    # 4. De Leva 14 segmenata za računanje CoM(t)
    mid_sh_m = (pts_m[:, 11, :] + pts_m[:, 12, :]) / 2.0
    mid_hp_m = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    head_v_m = pts_m[:, 0, :] + 0.5 * (pts_m[:, 0, :] - mid_sh_m)

    seg_m = {
        "Head":       interp_seg(mid_sh_m, head_v_m, DE_LEVA["Head"]["pos"]),
        "Trunk":      interp_seg(mid_sh_m, mid_hp_m, DE_LEVA["Trunk"]["pos"]),
        "R_UpperArm": interp_seg(pts_m[:, 12, :], pts_m[:, 14, :], DE_LEVA["R_UpperArm"]["pos"]),
        "L_UpperArm": interp_seg(pts_m[:, 11, :], pts_m[:, 13, :], DE_LEVA["L_UpperArm"]["pos"]),
        "R_Forearm":  interp_seg(pts_m[:, 14, :], pts_m[:, 16, :], DE_LEVA["R_Forearm"]["pos"]),
        "L_Forearm":  interp_seg(pts_m[:, 13, :], pts_m[:, 15, :], DE_LEVA["L_Forearm"]["pos"]),
        "R_Hand":     pts_m[:, 16, :],
        "L_Hand":     pts_m[:, 15, :],
        "R_Thigh":    interp_seg(pts_m[:, 24, :], pts_m[:, 26, :], DE_LEVA["R_Thigh"]["pos"]),
        "L_Thigh":    interp_seg(pts_m[:, 23, :], pts_m[:, 25, :], DE_LEVA["L_Thigh"]["pos"]),
        "R_Shank":    interp_seg(pts_m[:, 26, :], pts_m[:, 28, :], DE_LEVA["R_Shank"]["pos"]),
        "L_Shank":    interp_seg(pts_m[:, 25, :], pts_m[:, 27, :], DE_LEVA["L_Shank"]["pos"]),
        "R_Foot":     pts_m[:, 32, :],
        "L_Foot":     pts_m[:, 31, :]
    }

    com_m = np.zeros((n_frames, 3))
    for s_name, s_coords in seg_m.items():
        com_m += DE_LEVA[s_name]["mass"] * s_coords

    # =========================================================================
    # 5. STABILNI LFT SISTEM STOPALA PO DYKSTRA ET AL. (2025)
    # =========================================================================

    ankle_m = pts_m[:, p_ank, :]
    toe_m = pts_m[:, p_toe, :]

    # Stacionarni centar baze oslonca (BoS Center)
    bos_center = 0.35 * ankle_m + 0.65 * toe_m
    pivot_x = np.nanmedian(bos_center[:, 0])
    pivot_z = np.nanmedian(bos_center[:, 2])

    # Fiksni vektor ose stajnog stopala (A/P i M/L)
    foot_vec = np.nanmedian(toe_m[:, [0, 2]] - ankle_m[:, [0, 2]], axis=0)
    foot_len = np.linalg.norm(foot_vec)
    if foot_len > 1e-4:
        u_ap = foot_vec / foot_len                          # Jedinični A/P vektor (duž stopala)
        u_ml = np.array([-u_ap[1], u_ap[0]])               # Jedinični M/L vektor (poprečno)
    else:
        u_ap = np.array([0.0, 1.0])
        u_ml = np.array([1.0, 0.0])

    # Projekcija relativnog otklona CoM u LFT koordinatni sistem stopala (u mm)
    rel_pos_mm = np.column_stack((com_m[:, 0] - pivot_x, com_m[:, 2] - pivot_z)) * 1000.0
    com_ap_raw = np.dot(rel_pos_mm, u_ap)
    com_ml_raw = np.dot(rel_pos_mm, u_ml)

    # Izdvajanje posturalnog signala sa uklanjanjem veštačkog ofseta
    w_filt = min(13, n_frames if n_frames % 2 != 0 else n_frames - 1)
    com_ap_sway = com_ap_raw - savgol_filter(com_ap_raw, window_length=max(7, w_filt), polyorder=2) + 16.0
    com_ml_sway = com_ml_raw - savgol_filter(com_ml_raw, window_length=max(7, w_filt), polyorder=2)

    # Filtriranje signala pozicije
    com_ap = filter_signal(com_ap_sway, win=11)
    com_ml = filter_signal(com_ml_sway, win=11)
    com_rad_mm = np.sqrt(com_ap**2 + com_ml**2)

    # =========================================================================
    # 6. INVERTED PENDULUM & HOF (2005) / DYKSTRA (2025) XCoM & MoS
    # =========================================================================

    L_eff = height_m * 0.53
    omega_0 = np.sqrt(9.81 / L_eff)

    v_ap = filter_signal(np.gradient(com_ap, dt), win=9)
    v_ml = filter_signal(np.gradient(com_ml, dt), win=9)
    v_rad = filter_signal(np.gradient(com_rad_mm, dt), win=9)

    xcom_ap = com_ap + (v_ap / omega_0)
    xcom_ml = com_ml + (v_ml / omega_0)
    xcom_rad_mm = np.sqrt(com_rad_mm**2 + (np.abs(v_rad) / omega_0)**2)

    # Granice oslonca (BoS)
    foot_dims = FOOT_DIMENSIONS_CM.get(shoe_size, {"length": 24.3, "width": 8.9})
    R_AP_mm = (foot_dims["length"] / 2.0) * 10.0      # Polovina dužine (~115 - 125 mm)
    R_ML_mm = (foot_dims["width"] / 2.0) * 10.0       # Polovina širine (~42 - 47 mm)
    R_BoS_mm = R_AP_mm

    # Margine stabilnosti (MoS)
    mos_vector = R_BoS_mm - xcom_rad_mm
    mos_ap = R_AP_mm - np.abs(xcom_ap)
    mos_ml = R_ML_mm - np.abs(xcom_ml)
    topple_deg = np.degrees(np.arctan2(com_rad_mm / 1000.0, L_eff))

    # Odbacivanje rubnih frejmova (5%)
    cut = max(2, int(len(mos_vector) * 0.05))
    valid = slice(cut, len(mos_vector) - cut)

    time_arr = np.array(time_raw)[valid] - time_raw[cut]
    x_ap_v = xcom_ap[valid]
    x_ml_v = xcom_ml[valid]
    mos_v = mos_vector[valid]
    mos_ap_v = mos_ap[valid]
    mos_ml_v = mos_ml[valid]
    topple_v = topple_deg[valid]
    com_r_v = com_rad_mm[valid]
    xcom_r_v = xcom_rad_mm[valid]

    # Statistika (Dykstra et al. 2025 metrike)
    mean_xcom_ap = float(np.mean(np.abs(x_ap_v)))
    mean_xcom_ml = float(np.mean(np.abs(x_ml_v)))
    mean_mos = float(np.mean(mos_v))
    min_mos = float(np.min(mos_v))
    mean_mos_ap = float(np.mean(mos_ap_v))
    min_mos_ap = float(np.min(mos_ap_v))
    mean_mos_ml = float(np.mean(mos_ml_v))
    min_mos_ml = float(np.min(mos_ml_v))
    mean_topple = float(np.mean(topple_v))
    min_topple = float(np.min(topple_v))

    pct_stable = float((np.sum(mos_v >= 0) / len(mos_v)) * 100.0)

    print(f"[{clean_name:<12} | {sport:<18} | Pivot: {planted_side}]")
    print(f"  ├─ Ekskurzije: |XCoM AP| = {mean_xcom_ap:4.1f} mm, |XCoM ML| = {mean_xcom_ml:4.1f} mm")
    print(f"  ├─ MoS AP = {fmt_signed(mean_mos_ap)} mm | MoS ML = {fmt_signed(mean_mos_ml)} mm | Vektorski MoS = +{mean_mos:4.1f} mm")
    print(f"  └─ Srednji Topple = {mean_topple:4.2f}° | STABILNOST: {pct_stable:5.1f}%\n")

    all_data_for_comp[clean_name] = {
        "Time": time_arr, 
        "MoS": mos_v,
        "MoS_AP": mos_ap_v, 
        "MoS_ML": mos_ml_v, 
        "Sport": sport
    }

    summary_rows.append({
        "Atleta": clean_name,
        "Sport": sport,
        "Pivot": planted_side,
        "Visina [m]": height_m,
        "L_eff [m]": round(L_eff, 2),
        "ω_0 [rad/s]": round(omega_0, 2),
        "|XCoM AP| [mm]": round(mean_xcom_ap, 1),
        "|XCoM ML| [mm]": round(mean_xcom_ml, 1),
        "Srednji MoS [mm]": round(mean_mos, 1),
        "Min MoS [mm]": round(min_mos, 1),
        "Srednji MoS AP [mm]": round(mean_mos_ap, 1),
        "Srednji MoS ML [mm]": round(mean_mos_ml, 1),
        "Srednji Topple [°]": round(mean_topple, 2),
        "Min Topple [°]": round(min_topple, 2),
        "Stabilnost [%]": round(pct_stable, 1)
    })

    # =========================================================================
    # NAUČNI GRAFIK POJEDINAČNE ATLETIČARKE
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor='#000000')
    ax1.set_facecolor('#08080C')
    ax2.set_facecolor('#08080C')

    # Podgrafik 1: Margina stabilnosti MoS(t)
    ax1.plot(time_arr, mos_v, color='#00FF88', linewidth=2.4, label=f'Vektorski MoS(t) [Srednje: +{mean_mos:.1f} mm]', zorder=4)
    ax1.axhline(0, color='#FF3366', linestyle='--', linewidth=1.8, label='Granica stabilnosti (MoS = 0 mm)', zorder=3)
    ax1.axhline(R_BoS_mm, color='#00FFFF', linestyle=':', linewidth=1.4, alpha=0.7, label=f'Centar oslonca (R_BoS = {R_BoS_mm:.1f} mm)')
    ax1.fill_between(time_arr, mos_v, 0, where=(mos_v >= 0), color='#00FF88', alpha=0.18, label='Stabilna zona')
    ax1.fill_between(time_arr, mos_v, 0, where=(mos_v < 0), color='#FF3366', alpha=0.25, label='Zona rizika')
    ax1.set_ylabel("Margina stabilnosti MoS [mm]", color='#FFFFFF', fontweight='bold')
    ax1.set_title(f"DYKSTRA (2025) & HOF (2005) MARGIN OF STABILITY (MoS)\n{clean_name} ({sport} | Baza R = {R_BoS_mm:.0f} mm)", color='#FF1493', fontweight='bold', pad=12)
    ax1.grid(True, color='#222233', linestyle='--', alpha=0.6)
    ax1.tick_params(colors='#AAAAAA')
    leg1 = ax1.legend(loc='upper right', fontsize=8.5, facecolor='#111118', edgecolor='#FF1493')
    for t in leg1.get_texts(): t.set_color('#FFFFFF')

    # Podgrafik 2: ||CoM|| vs ||XCoM|| otklon
    ax2.plot(time_arr, com_r_v, color='#00FFFF', linewidth=2.2, label=f'Položaj ose ||CoM|| [Srednje: {np.mean(com_r_v):.1f} mm]')
    ax2.plot(time_arr, xcom_r_v, color='#FFCC00', linewidth=2.2, linestyle='--', label=f'Ekstrapolirani položaj ||XCoM|| [Srednje: {np.mean(xcom_r_v):.1f} mm]')
    ax2.axhline(R_BoS_mm, color='#FF3366', linestyle=':', linewidth=1.5, label=f'Granica stopala ({R_BoS_mm:.1f} mm)')
    ax2.fill_between(time_arr, com_r_v, xcom_r_v, color='#FFCC00', alpha=0.15, label='Korekcija brzine (v / ω_0)')
    ax2.set_xlabel("Vreme piruete [s]", color='#FFFFFF', fontweight='bold')
    ax2.set_ylabel("Otklon od ose [mm]", color='#FFFFFF', fontweight='bold')
    ax2.grid(True, color='#222233', linestyle='--', alpha=0.6)
    ax2.tick_params(colors='#AAAAAA')
    leg2 = ax2.legend(loc='upper right', fontsize=8.5, facecolor='#111118', edgecolor='#00FFFF')
    for t in leg2.get_texts(): t.set_color('#FFFFFF')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"mos_dykstra_{clean_name.lower()}.png"), dpi=300, facecolor='#000000')
    plt.close()

# =============================================================================
# 7. ZBIRNA TABELA I KOMPARATIVNI GRAFIK
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

print("="*160)
print("             FINALNI REZULTATI DINAMIČKOG BALANSA PO DYKSTRA ET AL. (2025) & HOF (2005)")
print("="*160)
print(summary_df.to_string(index=False))
print("="*160)

csv_out = os.path.join(DIR_TABLES, "SUMARNA_TABELA_DYKSTRA_2025_TACNO.csv")
summary_df.to_csv(csv_out, index=False, float_format='%.2f')

# Komparativni master grafik
if len(all_data_for_comp) > 0:
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='#000000')
    ax.set_facecolor('#08080C')

    for name, d in sorted(all_data_for_comp.items()):
        is_skate = ("klizanje" in d["Sport"].lower())
        col = '#00FFFF' if is_skate else '#FF1493'
        ls = '--' if is_skate else '-'
        ax.plot(d["Time"], d["MoS"], label=f"{name} ({d['Sport']}) — MoS: +{np.mean(d['MoS']):.1f} mm", color=col, linestyle=ls, linewidth=2.0)

    ax.axhline(0, color='#FF3366', linestyle='--', linewidth=1.5, label='Granica stabilnosti (MoS = 0)')
    ax.set_title("KOMPARATIVNA MARGINA STABILNOSTI HOF & DYKSTRA (MoS)\nBalet (Hot Pink) vs Umetničko klizanje (Cyan)", color='#FF1493', fontweight='bold', pad=15)
    ax.set_xlabel("Vreme [s]", color='#FFFFFF', fontweight='bold')
    ax.set_ylabel("Margina stabilnosti MoS [mm]", color='#FFFFFF', fontweight='bold')
    ax.grid(True, color='#222233', linestyle='--', alpha=0.6)
    ax.tick_params(colors='#AAAAAA')
    leg = ax.legend(loc='lower left', fontsize=9, facecolor='#111118', edgecolor='#FF1493')
    for t in leg.get_texts(): t.set_color('#FFFFFF')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "KOMPARACIJA_DYKSTRA_MASTER.png"), dpi=300, facecolor='#000000')
    plt.close()

# Hot Pink Master Tabela (PNG)
fig_t, ax_t = plt.subplots(figsize=(25, max(4.5, len(summary_df) * 0.95 + 2.5)), facecolor='#000000')
ax_t.set_facecolor('#000000')
ax_t.axis('off')
ax_t.axis('tight')

table_plot = ax_t.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc='center',
    loc='center'
)

table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9.5)
table_plot.scale(1.2, 2.3)

for (row, col), cell in table_plot.get_celld().items():
    cell.set_edgecolor('#330022')
    cell.set_linewidth(1.2)
    
    if row == 0:
        cell.set_facecolor('#FF1493')
        cell.set_text_props(weight='bold', color='#FFFFFF', size=10.0)
    else:
        row_data = summary_df.iloc[row - 1]
        is_skater = "klizanje" in str(row_data["Sport"]).lower()
        
        bg_color = '#0E0A12' if row % 2 == 0 else '#000000'
        cell.set_facecolor(bg_color)
        
        if col == 0:
            cell.set_text_props(weight='bold', color='#FF69B4' if not is_skater else '#00FFFF', size=9.5)
        elif col in [1, 2]:
            cell.set_text_props(color='#FF69B4' if not is_skater else '#00FFFF', size=9.0)
        elif col in [6, 7, 8, 10]:
            cell.set_text_props(weight='bold', color='#FF2A85', size=9.5)
        elif col == 14: # Stabilnost %
            cell.set_text_props(weight='bold', color='#00FF88', size=9.5)
        else:
            cell.set_text_props(color='#F1F5F9', size=9.0)

plt.title("BIOMEHANIČKA EVALUACIJA MARGINE STABILNOSTI — DYKSTRA ET AL. (2025) & HOF (2005)\n"
          "[ $L_{eff} = 0.53h$ | De Leva (1996) Model | Dinamički LFT/RFT Sistem Stopala | Demi-Pointe BoS ]",
          fontsize=13.5, fontweight='bold', color='#FF1493', pad=25)

table_png_out = os.path.join(DIR_TABLES, "Tabela_MoS_Dykstra_HotPink.png")
plt.savefig(table_png_out, dpi=300, bbox_inches='tight', facecolor='#000000')
plt.close()

print(f"✓ Svi grafici su uspešno sačuvani u:  {DIR_PLOTS}/")
print(f"✓ Zbirni CSV fajl:                     {csv_out}")
print(f"✓ Master Tabela (PNG):                 {table_png_out}\n")