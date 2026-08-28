import pandas as pd
import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. ANTROPOMETRIJSKI I FIZIČKI PARAMETRI
# ---------------------------------------------------------
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

input_dir = "konacne_koordinate"
output_coord_dir = "koordinate_sredjene"
output_table_folder = "tabele"
os.makedirs(output_coord_dir, exist_ok=True)
os.makedirs(output_table_folder, exist_ok=True)

FPS = 30
dt = 1.0 / FPS

def clean_and_interpolate_standard(df):
    """Proverena standardna interpolacija i čišćenje šuma"""
    df = df.copy()
    fps = float(df['FPS'].iloc[0]) if 'FPS' in df.columns and not df['FPS'].isna().all() else 30.0
    
    min_frame, max_frame = int(df['Frame'].min()), int(df['Frame'].max())
    all_frames = np.arange(min_frame, max_frame + 1)
    
    cleaned_groups = []
    for landmark_id in sorted(df['Landmark_ID'].unique()):
        group = df[df['Landmark_ID'] == landmark_id].copy()
        group = group.drop_duplicates(subset=['Frame'], keep='first').set_index('Frame')
        group = group.reindex(all_frames)
        group.index.name = 'Frame'
        group['Landmark_ID'] = landmark_id
        group['FPS'] = fps
        group['Time'] = group.index.to_numpy() / fps

        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = pd.to_numeric(group[col], errors='coerce')
            # Standardna linearna interpolacija sa popunjavanjem ivica
            vals = vals.interpolate(method='linear', limit_direction='both').ffill().bfill()
            group[col] = vals.values

        cleaned_groups.append(group.reset_index())
        
    return pd.concat(cleaned_groups, ignore_index=True).sort_values(['Frame', 'Landmark_ID']).reset_index(drop=True)

print(f"\nUčitavam i obrađujem fajlove iz: {input_dir}")
all_files = glob.glob(os.path.join(input_dir, "*_konacne.csv"))

table_rows = []
inertia_all_athletes = {}
trajectory_all_athletes = {}

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    athlete_key = None
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "Nepoznato"}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_key = key
            athlete_data = val
            break
            
    atype = athlete_data["type"]
    weight = athlete_data["weight"]
    
    # Standardna obrada i čuvanje u koordinate_sredjene
    df_clean = clean_and_interpolate_standard(df_raw)
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).replace("_konacne.csv", "").upper()
    
    coord_out_path = os.path.join(output_coord_dir, f"{clean_name.lower()}_sredjene.csv")
    df_clean.to_csv(coord_out_path, index=False)

    # Određivanje pivot noge
    foot_left = df_clean[df_clean['Landmark_ID'] == 31][['X_clean', 'Y_clean', 'Z_clean']].values
    foot_right = df_clean[df_clean['Landmark_ID'] == 32][['X_clean', 'Y_clean', 'Z_clean']].values
    move_left = np.sum(np.std(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    move_right = np.sum(np.std(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    
    planted_foot_id = 31 if move_left <= move_right else 32
    pivot_name = "Leva (31)" if planted_foot_id == 31 else "Desna (32)"

    frames_sorted = sorted(df_clean['Frame'].unique())
    angle_list, com_list, inertia_list = [], [], []

    for frame_idx in frames_sorted:
        f_df = df_clean[df_clean['Frame'] == frame_idx].set_index('Landmark_ID')
        if f_df.empty: continue
        frame_pts = {lm_id: f_df.loc[lm_id, ['X_clean', 'Y_clean', 'Z_clean']].values.astype(float) for lm_id in f_df.index if lm_id in f_df.index}
        
        if planted_foot_id in frame_pts:
            anchor = frame_pts[planted_foot_id]
            
            # Računanje pozicija segmenata u odnosu na pivot nogu u XZ ravni
            segments = {
                "Trunk": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 23, 24]], axis=0),
                "Head": frame_pts.get(0, anchor),
                "Thigh": np.mean([frame_pts.get(i, anchor) for i in [23, 24, 25, 26]], axis=0),
                "UpperArm": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 13, 14]], axis=0),
                "Forearm": np.mean([frame_pts.get(i, anchor) for i in [13, 14, 15, 16]], axis=0),
                "Hand": np.mean([frame_pts.get(i, anchor) for i in [15, 16]], axis=0),
                "Shank": np.mean([frame_pts.get(i, anchor) for i in [25, 26, 27, 28]], axis=0),
            }
            
            # Moment inercije (I = m * r^2 u XZ ravni)
            I_total = 0.0
            for seg_name, pos in segments.items():
                if seg_name in DE_LEVA_MASS_FEMALE:
                    r_sq = (pos[0] - anchor[0])**2 + (pos[2] - anchor[2])**2
                    I_total += DE_LEVA_MASS_FEMALE[seg_name] * weight * r_sq
            inertia_list.append(I_total)

            # Centar mase
            com = np.zeros(3); tw = 0.0
            for sn, sp in segments.items():
                if sn in DE_LEVA_MASS_FEMALE:
                    com += DE_LEVA_MASS_FEMALE[sn] * sp
                    tw += DE_LEVA_MASS_FEMALE[sn]
            if tw > 0: com /= tw
            com_list.append(com)
        else:
            inertia_list.append(0.0)
            com_list.append(np.zeros(3))

    # Proračun uglova i ugaonih brzina
    angle_list = []
    for frame_idx in frames_sorted:
        f_df = df_clean[df_clean['Frame'] == frame_idx].set_index('Landmark_ID')
        if 11 in f_df.index and 12 in f_df.index:
            p11 = f_df.loc[11, ['X_clean', 'Z_clean']].values.astype(float)
            p12 = f_df.loc[12, ['X_clean', 'Z_clean']].values.astype(float)
            dx, dz = p12[0] - p11[0], p12[1] - p11[1]
            angle_list.append(np.arctan2(dz, dx))
        else:
            angle_list.append(0.0)

    angles_raw = np.array(angle_list)
    # Unwrap uglova
    theta_unwrapped = np.unwrap(angles_raw)
    theta_smooth = savgol_filter(theta_unwrapped, window_length=9, polyorder=2)

    total_displacement_deg = np.degrees(np.abs(theta_smooth[-1] - theta_smooth[0]))
    total_rotations = total_displacement_deg / 360.0

    omega = np.gradient(theta_smooth, dt)
    omega_smooth = savgol_filter(omega, window_length=7, polyorder=2)
    max_omega_deg = np.max(np.abs(np.degrees(omega_smooth)))
    mean_omega_deg = np.mean(np.abs(np.degrees(omega_smooth)))

    alpha = np.gradient(omega_smooth, dt)
    max_alpha = np.max(np.abs(savgol_filter(alpha, window_length=9, polyorder=2)))

    com_arr = np.array(com_list)
    com_xz = np.sqrt(com_arr[:, 0]**2 + com_arr[:, 2]**2)
    com_fluctuation_cm = np.std(savgol_filter(com_xz, window_length=15, polyorder=3)) * 100.0

    inertia_arr = savgol_filter(np.array(inertia_list), window_length=11, polyorder=2)
    I_mean = np.mean(inertia_arr)
    KE_max = np.max(0.5 * inertia_arr * (omega_smooth ** 2))

    # Čuvanje za uporedne grafike
    inertia_all_athletes[clean_name] = inertia_arr
    trajectory_all_athletes[clean_name] = (com_arr[:, 0], com_arr[:, 2])

    table_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Pivot": pivot_name,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Broj okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(max_omega_deg, 1),
        "Srednja ω [°/s]": round(mean_omega_deg, 1),
        "Maks α [rad/s²]": round(max_alpha, 2),
        "I_mean [kg·m²]": round(I_mean, 3),
        "Max KE [J]": round(KE_max, 1),
        "CoM flukt. [cm]": round(com_fluctuation_cm, 2)
    })

summary_df = pd.DataFrame(table_rows)

print("\n\n" + "=" * 130)
print("REZULTATI FIZIČKE ANALIZE SA SREĐENIM KOORDINATAMA")
print("=" * 130)
print(summary_df.to_string(index=False))
print("=" * 130)

# ---------------------------------------------------------
# KREIRANJE UPOREDNIH GRAFIKA
# ---------------------------------------------------------
plt.figure(figsize=(12, 6), facecolor='#fff0f5')
ax = plt.gca()
ax.set_facecolor('#fffcfc')

for name, inertia_vals in inertia_all_athletes.items():
    time_axis = np.arange(len(inertia_vals)) * dt
    plt.plot(time_axis, inertia_vals, linewidth=2, label=name)

plt.title("Uporedni prikaz Momenta Inercije (I) kroz vreme za sve sportiste", fontsize=13, fontweight='bold', color='#c71585')
plt.xlabel("Vreme [s]", fontsize=11, fontweight='bold', color='#333333')
plt.ylabel("Moment inercije [kg·m²]", fontsize=11, fontweight='bold', color='#333333')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
inertia_plot_path = os.path.join(output_table_folder, "Uporedni_Moment_Inercije.png")
plt.savefig(inertia_plot_path, dpi=300, facecolor='#fff0f5')
plt.close()

# Grafik pogleda odozgo (XZ ravan)
plt.figure(figsize=(10, 8), facecolor='#fff0f5')
ax = plt.gca()
ax.set_facecolor('#fffcfc')

for name, (cx, cz) in trajectory_all_athletes.items():
    plt.plot(cx * 100, cz * 100, linewidth=2, label=f"{name}")

plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.title("Pogled odozgo (XZ ravan): Putanja centra mase i stabilnost", fontsize=13, fontweight='bold', color='#c71585')
plt.xlabel("X osmatranje [cm]", fontsize=11, fontweight='bold', color='#333333')
plt.ylabel("Z osmatranje [cm]", fontsize=11, fontweight='bold', color='#333333')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
traj_plot_path = os.path.join(output_table_folder, "Pogled_Odozgo_XZ_Ravan.png")
plt.savefig(traj_plot_path, dpi=300, facecolor='#fff0f5')
plt.close()

print(f"\n Grafici uspešno sačuvani u folder '{output_table_folder}':\n - Uporedni_Moment_Inercije.png\n - Pogled_Odozgo_XZ_Ravan.png\n")