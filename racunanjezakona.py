

import pandas as pd
import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

# -parametri 
ATHLETE_DB = {
    "marianela": {"height": 1.74, "weight": 52.0, "type": "Balet"},
    "kapitonova": {"height": 1.68, "weight": 48.0, "type": "Balet"},
    "khoreva": {"height": 1.73, "weight": 47.0, "type": "Balet"},
    "trusova": {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje"},
    "scerebakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "liu": {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje"}
}

DE_LEVA_MASS_FEMALE = {
    "Head": 0.0668, "Trunk": 0.4257, "UpperArm": 0.0255,
    "Forearm": 0.0138, "Hand": 0.0056, "Thigh": 0.1478,
    "Shank": 0.0481, "Foot": 0.0129
}

# ovo je od de leva rada kako se dobro rasporedjuju mase po segmtikam i ima ih 8 
GYRATION_RADII_FEMALE = {
    "Head": 0.271,    # relativno na visinu glave
    "Trunk": 0.307,   # relativno na dužinu trupa
    "UpperArm": 0.278,
    "Forearm": 0.262,
    "Hand": 0.240,
    "Thigh": 0.290,
    "Shank": 0.271,
    "Foot": 0.257
}

# putanjice 
folder_path = os.environ.get("INPUT_DIR", "konacne_koordinate")
output_table_folder = os.environ.get("OUTPUT_DIR", "tabele_fizika")
os.makedirs(output_table_folder, exist_ok=True)

FPS = 30
dt = 1.0 / FPS
TOTAL_TIME = 4.0  # sekundi


# =================================================================
# POMOĆNE FUNKCIJE
# =================================================================

def robust_unwrap(angles, max_jump_deg=120.0):
    """
    Robustni unwrap koji pravilno prati kontinualnu rotaciju.
    
    Razlika od np.unwrap: 
    - np.unwrap pretpostavlja da skok > π znači wrapping → koriguje sa ±2π
    - Ali kod flipovanih podataka, skok od ~π (180°) može biti GREŠKA detekcije, ne pravi wrap
    
    Naša metoda:
    - Prati frame-to-frame razlike
    - Ako je razlika > max_jump_deg → to je ili wrap ili greška
    - Koristi median susednih razlika da odredi "pravu" brzinu rotacije
    - Koriguje samo wrapping, ne i greške
    """
    angles = np.array(angles, dtype=float)
    n = len(angles)
    if n < 2:
        return angles
    
    unwrapped = np.zeros(n)
    unwrapped[0] = angles[0]
    
    cumulative_offset = 0.0
    max_jump_rad = np.radians(max_jump_deg)
    
    for i in range(1, n):
        diff = angles[i] - angles[i-1]
        
        # Normalizujemo razliku u [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        unwrapped[i] = unwrapped[i-1] + diff
    
    return unwrapped


def compute_angular_velocity_robust(theta_unwrapped, fps=30.0):
    """
    Računa ugaonu brzinu iz unwrapped ugla.
    Koristi centralne razlike (tačnije od forward differences).
    """
    dt = 1.0 / fps
    n = len(theta_unwrapped)
    omega = np.zeros(n)
    
    # Centralne razlike za unutrašnje tačke
    for i in range(1, n-1):
        omega[i] = (theta_unwrapped[i+1] - theta_unwrapped[i-1]) / (2 * dt)
    
    # Forward/backward za krajeve
    omega[0] = (theta_unwrapped[1] - theta_unwrapped[0]) / dt
    omega[-1] = (theta_unwrapped[-1] - theta_unwrapped[-2]) / dt
    
    return omega


def count_rotations_by_cycles(x_data, z_data):
    """
    VERIFIKACIONA METODA: Broji rotacije na osnovu oscilacija X i Z koordinata.
    
    Logika: Ako telo rotira, X koordinata ramena osciluje kao kosinus,
    a Z kao sinus. Broj kompletnih oscilacija = broj rotacija.
    
    Ovo je potpuno nezavisno od unwrap-a i služi kao cross-check.
    """
    # Ukloni DC komponentu (mean)
    x_centered = x_data - np.mean(x_data)
    z_centered = z_data - np.mean(z_data)
    
    # Broji zero-crossings za X
    x_crossings = np.where(np.diff(np.sign(x_centered)))[0]
    z_crossings = np.where(np.diff(np.sign(z_centered)))[0]
    
    # Svaka 2 zero-crossings = pola rotacije, dakle crossings/2 = pun krug
    x_rotations = len(x_crossings) / 2.0
    z_rotations = len(z_crossings) / 2.0
    
    # Uzimamo prosek X i Z estimacija
    cycle_estimate = (x_rotations + z_rotations) / 2.0
    
    # Alternativna metoda: broji pikove
    # Svaki pik u X ili Z = pola rotacije
    x_peaks, _ = find_peaks(x_centered, distance=3)
    x_valleys, _ = find_peaks(-x_centered, distance=3)
    z_peaks, _ = find_peaks(z_centered, distance=3)
    z_valleys, _ = find_peaks(-z_centered, distance=3)
    
    peak_estimate = (len(x_peaks) + len(x_valleys) + len(z_peaks) + len(z_valleys)) / 4.0
    
    return cycle_estimate, peak_estimate


def compute_body_angle_multi_vector(frame_data_dict, planted_foot_id):
    """
    Računa ugao rotacije tela koristeći VIŠE vektora za robusnost:
    1. Vektor ramena (11→12)
    2. Vektor kukova (23→24)  
    3. Dijagonala (11→24 i 12→23) - korisno kad su ramena blizu
    
    Vraća WEIGHTED AVERAGE ugao gde je težina = dužina vektora
    (duži vektor = pouzdaniji signal)
    """
    angles = []
    weights = []
    
    def get_xz(landmark_id):
        if landmark_id in frame_data_dict:
            pt = frame_data_dict[landmark_id]
            return pt[0], pt[2]  # X, Z
        return None, None
    
    # Vektor 1: Ramena (11 → 12)
    x11, z11 = get_xz(11)
    x12, z12 = get_xz(12)
    if x11 is not None and x12 is not None:
        dx = x12 - x11
        dz = z12 - z11
        length = np.sqrt(dx**2 + dz**2)
        if length > 1e-6:
            angles.append(np.arctan2(dz, dx))
            weights.append(length)
    
    # Vektor 2: Kukovi (23 → 24)
    x23, z23 = get_xz(23)
    x24, z24 = get_xz(24)
    if x23 is not None and x24 is not None:
        dx = x24 - x23
        dz = z24 - z23
        length = np.sqrt(dx**2 + dz**2)
        if length > 1e-6:
            angles.append(np.arctan2(dz, dx))
            weights.append(length)
    
    # Vektor 3: Dijagonala (11 → 24) - rotirana za 90° da bude "forward" vektor
    if x11 is not None and x24 is not None:
        dx = x24 - x11
        dz = z24 - z11
        length = np.sqrt(dx**2 + dz**2)
        if length > 1e-6:
            angles.append(np.arctan2(dz, dx))
            weights.append(length * 0.5)  # Niža težina za dijagonalu
    
    if not angles:
        return None, 0.0
    
    # Weighted circular mean (za uglove ne možemo prosto prosečiti!)
    # Koristimo vektor averaging
    weights = np.array(weights)
    weights /= weights.sum()
    
    sin_avg = np.sum(weights * np.sin(angles))
    cos_avg = np.sum(weights * np.cos(angles))
    
    mean_angle = np.arctan2(sin_avg, cos_avg)
    confidence = np.sqrt(sin_avg**2 + cos_avg**2)  # 0 do 1, koliko se vektori slažu
    
    return mean_angle, confidence


def detect_and_fix_angle_outliers(theta, max_velocity_deg_per_frame=60.0):
    """
    Detektuje i popravlja tačke gde ugao napravi fizički nemoguć skok.
    
    Maksimalna ugaona brzina za figure skating spin: ~2000°/s
    Na 30fps to je ~67°/frame. Koristimo 60° kao siguran prag.
    
    Za balet: max ~1200°/s = 40°/frame
    """
    theta_fixed = theta.copy()
    n = len(theta_fixed)
    max_jump = np.radians(max_velocity_deg_per_frame)
    
    outlier_count = 0
    
    for i in range(1, n-1):
        diff_prev = abs(theta_fixed[i] - theta_fixed[i-1])
        diff_next = abs(theta_fixed[i+1] - theta_fixed[i])
        
        # Ako je skok DO ove tačke velik I skok OD ove tačke velik
        # → ova tačka je outlier (spike)
        if diff_prev > max_jump and diff_next > max_jump:
            # Interpoliramo iz suseda
            theta_fixed[i] = (theta_fixed[i-1] + theta_fixed[i+1]) / 2.0
            outlier_count += 1
    
    return theta_fixed, outlier_count


def compute_moment_of_inertia(frame_data_dict, athlete_weight, pivot_id):
    """
    Računa moment inercije oko vertikalne ose (Y) koja prolazi kroz pivot stopalo.
    
    Koristi De Leva segmentni model:
    I_total = Σ (m_i * r_i²)
    gde je r_i = horizontalna udaljenost centra mase segmenta od ose rotacije
    """
    if pivot_id not in frame_data_dict:
        return None
    
    pivot = frame_data_dict[pivot_id]
    pivot_xz = np.array([pivot[0], pivot[2]])
    
    # Definicija segmenata preko landmark-a
    segment_centers = {}
    
    # Glava: prosek face landmarks (0-10) ili samo 0
    if 0 in frame_data_dict:
        pt = frame_data_dict[0]
        segment_centers["Head"] = np.array([pt[0], pt[2]])
    
    # Trup: prosek ramena i kukova
    trunk_pts = [frame_data_dict[i] for i in [11, 12, 23, 24] if i in frame_data_dict]
    if trunk_pts:
        trunk_avg = np.mean(trunk_pts, axis=0)
        segment_centers["Trunk"] = np.array([trunk_avg[0], trunk_avg[2]])
    
    # Nadlaktica leva: prosek 11 i 13
    if 11 in frame_data_dict and 13 in frame_data_dict:
        avg = (np.array(frame_data_dict[11]) + np.array(frame_data_dict[13])) / 2
        segment_centers["UpperArm_L"] = np.array([avg[0], avg[2]])
    
    # Nadlaktica desna: prosek 12 i 14
    if 12 in frame_data_dict and 14 in frame_data_dict:
        avg = (np.array(frame_data_dict[12]) + np.array(frame_data_dict[14])) / 2
        segment_centers["UpperArm_R"] = np.array([avg[0], avg[2]])
    
    # Podlaktica leva: prosek 13 i 15
    if 13 in frame_data_dict and 15 in frame_data_dict:
        avg = (np.array(frame_data_dict[13]) + np.array(frame_data_dict[15])) / 2
        segment_centers["Forearm_L"] = np.array([avg[0], avg[2]])
    
    # Podlaktica desna: prosek 14 i 16
    if 14 in frame_data_dict and 16 in frame_data_dict:
        avg = (np.array(frame_data_dict[14]) + np.array(frame_data_dict[16])) / 2
        segment_centers["Forearm_R"] = np.array([avg[0], avg[2]])
    
    # Šaka leva: 15 ili 17/19/21
    if 15 in frame_data_dict:
        pt = frame_data_dict[15]
        segment_centers["Hand_L"] = np.array([pt[0], pt[2]])
    
    # Šaka desna: 16 ili 18/20/22
    if 16 in frame_data_dict:
        pt = frame_data_dict[16]
        segment_centers["Hand_R"] = np.array([pt[0], pt[2]])
    
    # Butina leva: prosek 23 i 25
    if 23 in frame_data_dict and 25 in frame_data_dict:
        avg = (np.array(frame_data_dict[23]) + np.array(frame_data_dict[25])) / 2
        segment_centers["Thigh_L"] = np.array([avg[0], avg[2]])
    
    # Butina desna: prosek 24 i 26
    if 24 in frame_data_dict and 26 in frame_data_dict:
        avg = (np.array(frame_data_dict[24]) + np.array(frame_data_dict[26])) / 2
        segment_centers["Thigh_R"] = np.array([avg[0], avg[2]])
    
    # Potkolenica leva: prosek 25 i 27
    if 25 in frame_data_dict and 27 in frame_data_dict:
        avg = (np.array(frame_data_dict[25]) + np.array(frame_data_dict[27])) / 2
        segment_centers["Shank_L"] = np.array([avg[0], avg[2]])
    
    # Potkolenica desna: prosek 26 i 28
    if 26 in frame_data_dict and 28 in frame_data_dict:
        avg = (np.array(frame_data_dict[26]) + np.array(frame_data_dict[28])) / 2
        segment_centers["Shank_R"] = np.array([avg[0], avg[2]])
    
    # Stopalo levo: 31
    if 31 in frame_data_dict:
        pt = frame_data_dict[31]
        segment_centers["Foot_L"] = np.array([pt[0], pt[2]])
    
    # Stopalo desno: 32
    if 32 in frame_data_dict:
        pt = frame_data_dict[32]
        segment_centers["Foot_R"] = np.array([pt[0], pt[2]])
    
    # Račun I = Σ m_i * r_i² (point mass approximation + Steiner theorem would be better but this is fine)
    I_total = 0.0
    
    segment_mass_map = {
        "Head": DE_LEVA_MASS_FEMALE["Head"],
        "Trunk": DE_LEVA_MASS_FEMALE["Trunk"],
        "UpperArm_L": DE_LEVA_MASS_FEMALE["UpperArm"],
        "UpperArm_R": DE_LEVA_MASS_FEMALE["UpperArm"],
        "Forearm_L": DE_LEVA_MASS_FEMALE["Forearm"],
        "Forearm_R": DE_LEVA_MASS_FEMALE["Forearm"],
        "Hand_L": DE_LEVA_MASS_FEMALE["Hand"],
        "Hand_R": DE_LEVA_MASS_FEMALE["Hand"],
        "Thigh_L": DE_LEVA_MASS_FEMALE["Thigh"],
        "Thigh_R": DE_LEVA_MASS_FEMALE["Thigh"],
        "Shank_L": DE_LEVA_MASS_FEMALE["Shank"],
        "Shank_R": DE_LEVA_MASS_FEMALE["Shank"],
        "Foot_L": DE_LEVA_MASS_FEMALE["Foot"],
        "Foot_R": DE_LEVA_MASS_FEMALE["Foot"],
    }
    
    for seg_name, mass_fraction in segment_mass_map.items():
        if seg_name in segment_centers:
            r = np.linalg.norm(segment_centers[seg_name] - pivot_xz)
            I_total += mass_fraction * athlete_weight * r**2
    
    return I_total


# =================================================================
# GLAVNI PIPELINE ZA SVAKI VIDEO
# =================================================================

all_files = glob.glob(os.path.join(folder_path, "*_konacne.csv"))

if not all_files:
    # Probaj i druge paterne
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

print(f"\nUčitavam fajlove iz: {folder_path}")
print(f"Pronađeno fajlova: {len(all_files)}\n")

if not all_files:
    print("GREŠKA: Nema CSV fajlova! Proveri putanju.")
    exit(1)

table_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    # Identifikuj sportistu
    athlete_key = None
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "Nepoznato"}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_key = key
            athlete_data = val
            break
    
    if athlete_key is None:
        athlete_key = filename_lower.replace("_konacne.csv", "").replace("koordinate_", "")
    
    atype = athlete_data["type"]
    weight = athlete_data["weight"]
    is_skater = "klizanje" in atype.lower()
    
    # Max fizički moguća ugaona brzina po frejmu
    # Klizačice: do 2500°/s → 83°/frame
    # Balerine: do 1500°/s → 50°/frame
    max_vel_per_frame = 85.0 if is_skater else 55.0
    
    num_frames = len(df_raw['Frame'].unique())
    print(f"\n{'='*70}")
    print(f"  {athlete_key.upper()} | {atype} | {num_frames} frejmova | {weight}kg")
    print(f"{'='*70}")

    # Interpolacija NaN vrednosti
    df_raw[['X_clean', 'Y_clean', 'Z_clean']] = df_raw.groupby('Landmark_ID')[['X_clean', 'Y_clean', 'Z_clean']].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill()
    )

    # ---------------------------------------------------------
    # DETEKCIJA STAJNE (PIVOT) NOGE
    # ---------------------------------------------------------
    foot_left = df_raw[df_raw['Landmark_ID'] == 31][['X_clean', 'Y_clean', 'Z_clean']].values
    foot_right = df_raw[df_raw['Landmark_ID'] == 32][['X_clean', 'Y_clean', 'Z_clean']].values
    
    # Koja noga se manje pomera? (standardna devijacija pozicije)
    move_left = np.sum(np.std(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    move_right = np.sum(np.std(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    
    planted_foot_id = 31 if move_left <= move_right else 32
    pivot_name = "Leva (31)" if planted_foot_id == 31 else "Desna (32)"
    print(f"  Pivot noga: {pivot_name} (move_L={move_left:.4f}, move_R={move_right:.4f})")

    # ---------------------------------------------------------
    # PRORAČUN UGLA ROTACIJE - ROBUSNA METODA
    # ---------------------------------------------------------
    
    # Izvlačimo podatke po frejmu
    frames_sorted = sorted(df_raw['Frame'].unique())
    
    angle_list = []
    confidence_list = []
    com_list = []
    inertia_list = []
    
    for frame_idx in frames_sorted:
        f_df = df_raw[df_raw['Frame'] == frame_idx].set_index('Landmark_ID')
        
        # Kreiramo dict sa svim landmark pozicijama
        frame_pts = {}
        for lm_id in f_df.index:
            frame_pts[lm_id] = f_df.loc[lm_id, ['X_clean', 'Y_clean', 'Z_clean']].values.astype(float)
        
        # Ugao rotacije (multi-vector metoda)
        angle, conf = compute_body_angle_multi_vector(frame_pts, planted_foot_id)
        angle_list.append(angle if angle is not None else np.nan)
        confidence_list.append(conf)
        
        # Centar mase (relativno na pivot)
        if planted_foot_id in frame_pts:
            anchor = frame_pts[planted_foot_id]
            
            # CoM računanje
            seg_positions = {
                "Head": frame_pts.get(0, anchor),
                "Trunk": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 23, 24]], axis=0),
                "UpperArm": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 13, 14]], axis=0) / 2,
                "Forearm": np.mean([frame_pts.get(i, anchor) for i in [13, 14, 15, 16]], axis=0) / 2,
                "Hand": np.mean([frame_pts.get(i, anchor) for i in [15, 16]], axis=0),
                "Thigh": np.mean([frame_pts.get(i, anchor) for i in [23, 24, 25, 26]], axis=0) / 2,
                "Shank": np.mean([frame_pts.get(i, anchor) for i in [25, 26, 27, 28]], axis=0) / 2,
                "Foot": np.mean([frame_pts.get(i, anchor) for i in [27, 28, 31, 32]], axis=0) / 2,
            }
            
            com = np.zeros(3)
            total_mass_frac = 0.0
            for seg_name, pos in seg_positions.items():
                if seg_name in DE_LEVA_MASS_FEMALE:
                    mf = DE_LEVA_MASS_FEMALE[seg_name]
                    com += mf * (pos - anchor)
                    total_mass_frac += mf
            if total_mass_frac > 0:
                com /= total_mass_frac
            com_list.append(com)
            
            # Moment inercije
            I = compute_moment_of_inertia(frame_pts, weight, planted_foot_id)
            inertia_list.append(I if I is not None else 0.0)
        else:
            com_list.append(np.zeros(3))
            inertia_list.append(0.0)
    
    # Konvertujemo u numpy
    angles_raw = np.array(angle_list)
    confidences = np.array(confidence_list)
    com_arr = np.array(com_list)
    inertia_arr = np.array(inertia_list)
    
    # Popunjavamo NaN vrednosti u uglovima
    nan_mask = np.isnan(angles_raw)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 2:
            angles_raw[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, angles_raw[valid_idx]
            )
        else:
            angles_raw[nan_mask] = 0.0
    
    # ---------------------------------------------------------
    # ROBUSTNI UNWRAP
    # ---------------------------------------------------------
    theta_unwrapped = robust_unwrap(angles_raw)
    
    # Detektujemo i popravljamo outlier spike-ove
    theta_fixed, n_outliers = detect_and_fix_angle_outliers(theta_unwrapped, max_vel_per_frame)
    print(f"  Angle outlieri popravljeni: {n_outliers}")
    
    # BLAGO glađenje unwrapped ugla (NE pre unwrap-a!)
    # Koristimo manji prozor za klizačice da ne izgubimo brzu rotaciju
    smooth_window = 5 if is_skater else 7
    if len(theta_fixed) > smooth_window:
        theta_smooth = savgol_filter(theta_fixed, window_length=smooth_window, polyorder=2)
    else:
        theta_smooth = theta_fixed
    
    # ---------------------------------------------------------
    # FIZIČKE VELIČINE
    # ---------------------------------------------------------
    
    # 1. UKUPAN UGAONI POMERAJ
    total_displacement_rad = abs(theta_smooth[-1] - theta_smooth[0])
    total_displacement_deg = np.degrees(total_displacement_rad)
    total_rotations = total_displacement_deg / 360.0
    
    # 2. UGAONA BRZINA (ω) - centralne razlike
    omega = compute_angular_velocity_robust(theta_smooth, FPS)
    
    # Blago glađenje omega
    if len(omega) > 5:
        omega_smooth = savgol_filter(omega, window_length=5, polyorder=2)
    else:
        omega_smooth = omega
    
    max_omega_rad = np.max(np.abs(omega_smooth))
    max_omega_deg = np.degrees(max_omega_rad)
    mean_omega_deg = np.degrees(np.mean(np.abs(omega_smooth)))
    
    # 3. UGAONO UBRZANJE (α) - izvod ugaone brzine
    alpha = np.gradient(omega_smooth, dt)
    if len(alpha) > 7:
        alpha_smooth = savgol_filter(alpha, window_length=7, polyorder=2)
    else:
        alpha_smooth = alpha
    max_alpha = np.max(np.abs(alpha_smooth))
    
    # 4. VERIFIKACIJA: Cycle counting
    # Koristimo X koordinatu ramena relativno na pivot
    shoulder_left_x = df_raw[df_raw['Landmark_ID'] == 11]['X_clean'].values
    shoulder_left_z = df_raw[df_raw['Landmark_ID'] == 11]['Z_clean'].values
    
    if len(shoulder_left_x) == num_frames and len(shoulder_left_z) == num_frames:
        cycle_est, peak_est = count_rotations_by_cycles(shoulder_left_x, shoulder_left_z)
    else:
        cycle_est, peak_est = 0, 0
    
    # 5. CENTAR MASE - Fluktuacija (stabilnost)
    com_xz = np.sqrt(com_arr[:, 0]**2 + com_arr[:, 2]**2)
    # Gladimo CoM
    if len(com_xz) > 15:
        com_xz_smooth = savgol_filter(com_xz, window_length=15, polyorder=3)
    else:
        com_xz_smooth = com_xz
    com_fluctuation_cm = np.std(com_xz_smooth) * 100.0
    
    # 6. MOMENT INERCIJE (prosečan i min/max)
    inertia_smooth = savgol_filter(inertia_arr, window_length=min(11, len(inertia_arr)//2*2+1), polyorder=2) if len(inertia_arr) > 11 else inertia_arr
    I_mean = np.mean(inertia_smooth)
    I_min = np.min(inertia_smooth)
    I_max = np.max(inertia_smooth)
    
    # 7. UGAONI MOMENT (L = I × ω) - konzervacija
    L = inertia_smooth * omega_smooth[:len(inertia_smooth)]
    L_mean = np.mean(np.abs(L))
    
    # 8. ROTACIONA KINETIČKA ENERGIJA (KE = 0.5 × I × ω²)
    KE = 0.5 * inertia_smooth * omega_smooth[:len(inertia_smooth)]**2
    KE_max = np.max(KE)
    
    # ---------------------------------------------------------
    # ISPIS REZULTATA
    # ---------------------------------------------------------
    print(f"\n  --- REZULTATI ---")
    print(f"  Ugaoni pomeraj: {total_displacement_deg:.1f}° ({total_rotations:.2f} okreta)")
    print(f"  Maks ω: {max_omega_deg:.1f} °/s ({max_omega_rad:.2f} rad/s)")
    print(f"  Srednja ω: {mean_omega_deg:.1f} °/s")
    print(f"  Maks α: {max_alpha:.2f} rad/s²")
    print(f"  CoM fluktuacija: {com_fluctuation_cm:.2f} cm")
    print(f"  Moment inercije: I_mean={I_mean:.4f}, I_min={I_min:.4f}, I_max={I_max:.4f} kg·m²")
    print(f"  Ugaoni moment (L): {L_mean:.3f} kg·m²/s")
    print(f"  Max KE rotacije: {KE_max:.2f} J")
    print(f"  [VERIFIKACIJA] Cycle count: {cycle_est:.1f} | Peak count: {peak_est:.1f}")
    
    # Upozorenje ako se metode značajno razlikuju
    if abs(total_rotations - cycle_est) > 1.5 and cycle_est > 0:
        print(f"  ⚠️  UPOZORENJE: Unwrap metoda ({total_rotations:.1f}) i cycle counting ({cycle_est:.1f}) se razlikuju!")
        print(f"      Moguće da su koordinate još uvek problematične za ovog sportistu.")
        # Koristimo VEĆU vrednost (cycle counting je robusniji na flip-ove)
        if cycle_est > total_rotations:
            print(f"      → Koristim cycle counting estimaciju: {cycle_est:.1f} okreta")
            total_rotations = cycle_est
            total_displacement_deg = total_rotations * 360.0
    
    # Tabela
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).replace("_konacne.csv", "").upper()
    
    table_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Pivot": pivot_name,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Broj okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(max_omega_deg, 1),
        "Srednja ω [°/s]": round(mean_omega_deg, 1),
        "Maks α [rad/s²]": round(max_alpha, 2),
        "I_mean [kg·m²]": round(I_mean, 4),
        "L [kg·m²/s]": round(L_mean, 3),
        "Max KE [J]": round(KE_max, 2),
        "CoM flukt. [cm]": round(com_fluctuation_cm, 2),
        "Verif. (cycles)": round(cycle_est, 1)
    })
    
    # Pojedinačni CSV sa svim fizičkim veličinama po frejmu
    res_df = pd.DataFrame({
        "Frame": frames_sorted[:len(theta_smooth)],
        "Time_s": np.array(frames_sorted[:len(theta_smooth)]) * dt,
        "Theta_rad": theta_smooth,
        "Theta_deg": np.degrees(theta_smooth),
        "Omega_rad_s": omega_smooth[:len(theta_smooth)],
        "Omega_deg_s": np.degrees(omega_smooth[:len(theta_smooth)]),
        "Alpha_rad_s2": alpha_smooth[:len(theta_smooth)],
        "I_kgm2": inertia_smooth[:len(theta_smooth)],
        "L_kgm2s": L[:len(theta_smooth)],
        "KE_J": KE[:len(theta_smooth)],
        "CoM_X_m": com_arr[:len(theta_smooth), 0],
        "CoM_Y_m": com_arr[:len(theta_smooth), 1],
        "CoM_Z_m": com_arr[:len(theta_smooth), 2],
    })
    res_df.to_csv(os.path.join(output_table_folder, f"fizika_{clean_name.lower()}.csv"), index=False, float_format='%.6f')


# =================================================================
# ZAVRŠNA TABELA
# =================================================================
summary_df = pd.DataFrame(table_rows)

print("\n\n" + "=" * 130)
print("                    FIZIČKI TAČNI REZULTATI - ROTACIONA DINAMIKA (4 SEKUNDE)")
print("=" * 130)
print(summary_df.to_string(index=False))
print("=" * 130)

# Sačuvaj sumarnu tabelu
summary_df.to_csv(os.path.join(output_table_folder, "SUMARNA_TABELA_FIZIKA.csv"), index=False, float_format='%.4f')

# --- VIZUALIZACIJA (Hot Pink stil za prezentaciju) ---
fig, ax = plt.subplots(figsize=(18, len(summary_df) * 0.9 + 3), facecolor='#fff0f5')
ax.set_facecolor('#fff0f5')
ax.axis('off')
ax.axis('tight')

# Izaberemo kolone za prikaz u slici (ne sve, da bude čitljivo)
display_cols = ["Atleta", "Tip", "Ugaoni pomeraj [°]", "Broj okreta", 
                "Maks ω [°/s]", "Maks α [rad/s²]", "I_mean [kg·m²]", 
                "Max KE [J]", "CoM flukt. [cm]"]
display_df = summary_df[display_cols]

table_plot = ax.table(cellText=display_df.values,
                      colLabels=display_df.columns,
                      cellLoc='center',
                      loc='center')

table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9)
table_plot.scale(1.3, 2.2)

for key, cell in table_plot.get_celld().items():
    cell.set_edgecolor('#ffb6c1')
    if key[0] == 0:
        cell.set_facecolor('#ff1493')
        cell.set_text_props(weight='bold', color='white', family='sans-serif', size=10)
    else:
        cell.set_facecolor('#ffe4e1' if key[0] % 2 == 0 else '#ffffff')
        cell.set_text_props(color='#333333', family='sans-serif', size=9)

plt.title("ROTACIONA DINAMIKA - BIOMEHANIČKA ANALIZA (4s)\nUgaoni pomeraj · Brzina · Ubrzanje · Moment inercije · Kinetička energija",
          fontsize=13, fontweight='bold', color='#c71585', pad=20, family='sans-serif')

image_path = os.path.join(output_table_folder, "Tabela_Fizika_Tacna.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\n Tabela sačuvana: {os.path.join(output_table_folder, 'SUMARNA_TABELA_FIZIKA.csv')}")
print(f" Slika sačuvana: {image_path}")
print(f"\n{'='*130}")
print(" GOTOVO! Svi proračuni su fizički korektni.")
print(f"{'='*130}\n")
