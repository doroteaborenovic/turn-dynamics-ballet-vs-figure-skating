import pandas as pd
import glob
import os
import numpy as np
import warnings
from scipy.signal import butter, filtfilt, savgol_filter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. ANTROPOMETRIJSKI I FIZIČKI PARAMETRI
# ---------------------------------------------------------
ATHLETE_DB = {
    "marianela": {"height": 1.74, "weight": 52.0, "type": "balet"},
    "kapitonova": {"height": 1.68, "weight": 48.0, "type": "balet"},
    "khoreva": {"height": 1.73, "weight": 47.0, "type": "balet"},
    "trusova": {"height": 1.66, "weight": 50.0, "type": "klizanje"},
    "scerebakova": {"height": 1.61, "weight": 42.0, "type": "klizanje"},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "klizanje"},
    "liu": {"height": 1.58, "weight": 45.0, "type": "klizanje"}
}

DE_LEVA_MASS_FEMALE = {
    "Head": 0.0668, "Trunk": 0.4257, "UpperArm": 0.0255,
    "Forearm": 0.0138, "Hand": 0.0056, "Thigh": 0.1478,
    "Shank": 0.0481, "Foot": 0.0129
}

# ---------------------------------------------------------
# 2. DEFINISANJE 5 DUŽINA STOPALA (BoS eksperiment)
# ---------------------------------------------------------
# Dužine u metrima: 22cm, 23cm, 24cm, 25cm, 26cm (Ekvivalenti EU obuće ~35 do ~40)
# Poluprečnik baze oslonca (R) je polovina dužine.
FOOT_SIZES_M = [0.22, 0.23, 0.24, 0.25, 0.26]
FOOT_RADII = [l / 2.0 for l in FOOT_SIZES_M]

folder_path = r"C:\Users\PC\Videos\Screen Recordings\processed_coordinates"
all_files = glob.glob(os.path.join(folder_path, "koordinate*.csv"))

output_cnn_folder = r"C:\Users\PC\gitara\projekat.fizika\CNN_Spremno"
os.makedirs(output_cnn_folder, exist_ok=True)

TARGET_FRAMES = 300
FPS = 30
dt = 1.0 / FPS
g_const = 9.81

# ---------------------------------------------------------
# FILTERI PRILAGOĐENI FIZICI ROTACIJE
# ---------------------------------------------------------
def smooth_vectors(data):
    """Blago peglanje vektora ramena/kukova da ne ubijemo brze rotacije (klizanje 5Hz)"""
    return savgol_filter(data, window_length=7, polyorder=2, axis=0)

def smooth_com(data):
    """Jako peglanje Centra Mase jer on mora biti stabilan. Ubija MediaPipe Z-drhtanje."""
    return savgol_filter(data, window_length=21, polyorder=3, axis=0)

print(f"\n{'Fajl':<25} | {'Krugovi':<7} | {'Nagib':<7} | {'Ubrzanje':<8} | {'MoS(22cm)':<9} | {'MoS(26cm)':<9}")
print("-" * 80)

for file in all_files:
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "nepoznato"}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_data = val
            break
            
    h = athlete_data["height"]
    m = athlete_data["weight"]

    # Interpolacija nestalih tačaka
    df_raw.loc[df_raw['Visibility'] < 0.5, ['X_world', 'Y_world', 'Z_world']] = np.nan
    df_raw[['X_world', 'Y_world', 'Z_world']] = df_raw.groupby('Landmark_ID')[['X_world', 'Y_world', 'Z_world']].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill()
    )

    # Identifikacija stajne noge (ona koja je najniža u proseku - najveće Y)
    avg_y_31 = df_raw[df_raw['Landmark_ID'] == 31]['Y_world'].mean()
    avg_y_32 = df_raw[df_raw['Landmark_ID'] == 32]['Y_world'].mean()
    planted_foot_id = 31 if avg_y_31 > avg_y_32 else 32

    original_frames = df_raw['Frame'].max() + 1
    new_data = []

    for lm_id in range(33):
        lm_df = df_raw[df_raw['Landmark_ID'] == lm_id].sort_values('Frame')
        if len(lm_df) == 0: continue
        
        old_indices = lm_df['Frame'].values
        new_indices = np.linspace(0, original_frames - 1, TARGET_FRAMES)
        
        x_w = np.interp(new_indices, old_indices, lm_df['X_world'].values)
        y_w = np.interp(new_indices, old_indices, lm_df['Y_world'].values)
        z_w = np.interp(new_indices, old_indices, lm_df['Z_world'].values)
        
        # Blago SavGol filtriranje celog tela
        x_w = savgol_filter(x_w, 9, 2)
        y_w = savgol_filter(y_w, 9, 2)
        z_w = savgol_filter(z_w, 9, 2)
        
        for f_new, xw, yw, zw in zip(range(TARGET_FRAMES), x_w, y_w, z_w):
            new_data.append({"Frame": f_new, "Landmark_ID": lm_id, "X_world": xw, "Y_world": yw, "Z_world": zw})
            
    df = pd.DataFrame(new_data)
    frame_results = []

    # Liste za vektore rotacije
    dx_list, dz_list = [], []
    com_rel_list = []
    l_pend_list = []

    for f in range(TARGET_FRAMES):
        f_df = df[df['Frame'] == f].set_index('Landmark_ID')
        def get_pt(idx): return f_df.loc[idx, ['X_world', 'Y_world', 'Z_world']].values.astype(float)

        anchor = get_pt(planted_foot_id)
        def rel_pt(idx): return get_pt(idx) - anchor

        # Segmenti i mase
        segments = {
            "Trunk": np.mean([rel_pt(11), rel_pt(12), rel_pt(23), rel_pt(24)], axis=0),
            "Head": np.mean([rel_pt(i) for i in range(11)], axis=0),
            "Thigh": 0.5 * (np.mean([rel_pt(23), rel_pt(25)], axis=0) + np.mean([rel_pt(24), rel_pt(26)], axis=0)),
            # Uprošćeni ostali segmenti (za stabilnost)
            "UpperArm": 0.5 * (rel_pt(11) + rel_pt(12)),
            "Forearm": 0.5 * (rel_pt(13) + rel_pt(14)),
            "Hand": 0.5 * (rel_pt(15) + rel_pt(16)),
            "Shank": 0.5 * (rel_pt(25) + rel_pt(26)),
            "Foot": 0.5 * (rel_pt(27) + rel_pt(28))
        }

        # CoM
        com_rel = np.zeros(3)
        for seg_name, p_seg in segments.items():
            com_rel += DE_LEVA_MASS_FEMALE[seg_name] * p_seg
        com_rel_list.append(com_rel)

        # Klatno
        l_pendulum = max(np.linalg.norm(com_rel), 0.3)
        l_pend_list.append(l_pendulum)

        # Vektori za rotaciju (Ramena + Kukovi)
        dx_sh = rel_pt(12)[0] - rel_pt(11)[0]
        dz_sh = rel_pt(12)[2] - rel_pt(11)[2]
        dx_hip = rel_pt(24)[0] - rel_pt(23)[0]
        dz_hip = rel_pt(24)[2] - rel_pt(23)[2]
        
        dx_list.append((dx_sh + dx_hip) / 2.0)
        dz_list.append((dz_sh + dz_hip) / 2.0)

    com_rel_arr = np.array(com_rel_list)
    
    # PEGLANJE Z-OSE (Ovo je rešenje za pad MediaPipe dubine)
    com_rel_arr[:, 0] = smooth_com(com_rel_arr[:, 0]) # X
    com_rel_arr[:, 2] = smooth_com(com_rel_arr[:, 2]) # Z

    dx_arr = smooth_vectors(np.array(dx_list))
    dz_arr = smooth_vectors(np.array(dz_list))
    
    # Računanje ugla ODPORNO NA GIMBAL LOCK
    theta_rot = np.unwrap(np.arctan2(dz_arr, dx_arr))
    # Ako ugao ima šum, provlačimo ga kroz blagi SavGol
    theta_rot = savgol_filter(theta_rot, window_length=11, polyorder=2)
    
    total_rotations = abs(theta_rot[-1] - theta_rot[0]) / (2 * np.pi)

    # Pravilna kinematička derivacija
    omega = savgol_filter(theta_rot, window_length=11, polyorder=2, deriv=1, delta=dt)
    alpha = savgol_filter(theta_rot, window_length=11, polyorder=2, deriv=2, delta=dt)

    res_df = pd.DataFrame({
        "Frame": range(TARGET_FRAMES),
        "Theta": theta_rot, "Omega": omega, "Alpha": alpha,
        "CoM_X_rel": com_rel_arr[:, 0], "CoM_Z_rel": com_rel_arr[:, 2],
        "L_pendulum": l_pend_list
    })

    # HOFOV MODEL XCoM
    vCoM_X = savgol_filter(res_df['CoM_X_rel'], window_length=15, polyorder=2, deriv=1, delta=dt)
    vCoM_Z = savgol_filter(res_df['CoM_Z_rel'], window_length=15, polyorder=2, deriv=1, delta=dt)
    
    res_df['omega_0'] = np.sqrt(g_const / res_df['L_pendulum'])
    res_df['XCoM_X'] = res_df['CoM_X_rel'] + (vCoM_X / res_df['omega_0'])
    res_df['XCoM_Z'] = res_df['CoM_Z_rel'] + (vCoM_Z / res_df['omega_0'])
    
    dist_to_foot = np.sqrt(res_df['XCoM_X']**2 + res_df['XCoM_Z']**2)
    
    # TILT CoM-a (PRAVI INDIKATOR STABILNOSTI!)
    r_com_f = np.sqrt(res_df['CoM_X_rel']**2 + res_df['CoM_Z_rel']**2)
    res_df['Tilt_deg'] = np.degrees(np.arcsin(np.clip(r_com_f / res_df['L_pendulum'], 0.0, 1.0)))

    # RAČUNANJE MoS ZA 5 DUŽINA STOPALA (Dodato po tvom zahtevu)
    # Svaka veličina obuće dobija svoju kolonu za CNN!
    for i, size_cm in enumerate([22, 23, 24, 25, 26]):
        R = FOOT_RADII[i]
        res_df[f'MoS_{size_cm}cm'] = R - dist_to_foot
        
        # Opciono: Faktor sigurnosti za svaki broj (možeš ubaciti u mrežu)
        theta_crit = np.degrees(np.arcsin(R / res_df['L_pendulum'].mean()))
        res_df[f'SF_{size_cm}cm'] = theta_crit / np.where(res_df['Tilt_deg'] < 0.1, 0.1, res_df['Tilt_deg'])

    res_df['Height'] = h
    res_df['Mass'] = m

    # SVE SE ČUVA U CSV ZA CNN
    save_path = os.path.join(output_cnn_folder, f"fizika_{os.path.basename(file)}")
    res_df.to_csv(save_path, index=False)

    avg_tilt = res_df['Tilt_deg'].mean()
    max_alpha = res_df['Alpha'].abs().max()
    min_mos_22 = res_df['MoS_22cm'].min()
    min_mos_26 = res_df['MoS_26cm'].min()

    print(f"{os.path.basename(file)[:24]:<25} | {total_rotations:.1f} kr | {avg_tilt:.1f}°   | {max_alpha:.1f} r/s2 | {min_mos_22:+.4f} m | {min_mos_26:+.4f} m")

print("-" * 80)
print(f"Svi podaci, ZAJEDNO SA 5 DUŽINA STOPALA, čuvaju se u:\n{output_cnn_folder}")