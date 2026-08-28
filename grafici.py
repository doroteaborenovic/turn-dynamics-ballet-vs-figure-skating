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

input_coord_dir = "konacne_koordinate"
if not os.path.exists(input_coord_dir):
    input_coord_dir = "koordinate_sredjene"

output_folder = "tabele_fizika"
os.makedirs(output_folder, exist_ok=True)

all_files = glob.glob(os.path.join(input_coord_dir, "*.csv"))
FPS = 30
dt = 1.0 / FPS

print(f"\nUčitavam podatke za ispravljenu analizu iz: {input_coord_dir}")
print(f"Pronađeno fajlova: {len(all_files)}\n")

physics_results_dict = {}

for file in sorted(all_files):
    df = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    athlete_key = None
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "Nepoznato"}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_key = key
            athlete_data = val
            break
            
    weight = athlete_data["weight"]
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).replace("_konacne.csv", "").replace(".csv", "").upper()

    # Određivanje pivot noge
    foot_left = df[df['Landmark_ID'] == 31][['X_clean', 'Y_clean', 'Z_clean']].values if 'Landmark_ID' in df.columns else np.array([])
    foot_right = df[df['Landmark_ID'] == 32][['X_clean', 'Y_clean', 'Z_clean']].values if 'Landmark_ID' in df.columns else np.array([])
    move_left = np.sum(np.std(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    move_right = np.sum(np.std(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    planted_foot_id = 31 if move_left <= move_right else 32

    frames_sorted = sorted(df['Frame'].unique())
    inertia_list, time_list = [], []
    vector_list = []

    for frame_idx in frames_sorted:
        f_df = df[df['Frame'] == frame_idx].set_index('Landmark_ID')
        if f_df.empty: continue
        
        frame_pts = {lm_id: f_df.loc[lm_id, ['X_clean', 'Y_clean', 'Z_clean']].values.astype(float) for lm_id in f_df.index if lm_id in f_df.index}
        curr_time = f_df['Time'].iloc[0] if 'Time' in f_df.columns else frame_idx / FPS
        
        # Vektor ramena (11 do 12) u XZ ravni za praćenje rotacije
        if 11 in frame_pts and 12 in frame_pts:
            dx = frame_pts[12][0] - frame_pts[11][0]
            dz = frame_pts[12][2] - frame_pts[11][2]
            vector_list.append((dx, dz))
        else:
            vector_list.append((0.0, 0.0))

        if planted_foot_id in frame_pts:
            anchor = frame_pts[planted_foot_id]
            segments = {
                "Trunk": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 23, 24]], axis=0),
                "Head": frame_pts.get(0, anchor),
                "Thigh": np.mean([frame_pts.get(i, anchor) for i in [23, 24, 25, 26]], axis=0),
                "UpperArm": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 13, 14]], axis=0),
                "Forearm": np.mean([frame_pts.get(i, anchor) for i in [13, 14, 15, 16]], axis=0),
                "Hand": np.mean([frame_pts.get(i, anchor) for i in [15, 16]], axis=0),
                "Shank": np.mean([frame_pts.get(i, anchor) for i in [25, 26, 27, 28]], axis=0),
            }
            
            I_total = 0.0
            for seg_name, pos in segments.items():
                if seg_name in DE_LEVA_MASS_FEMALE:
                    r_sq = (pos[0] - anchor[0])**2 + (pos[2] - anchor[2])**2
                    I_total += DE_LEVA_MASS_FEMALE[seg_name] * weight * r_sq
                    
            if np.isfinite(I_total):
                inertia_list.append(I_total)
                time_list.append(curr_time)

    # ISPRAVLJENO RAČUNANJE UGAONE BRZINE PREKO DELTA UGLOVA (Nema minusa!)
    omega_deg = np.zeros(len(vector_list))
    alpha_smooth = np.zeros(len(vector_list))
    
    if len(vector_list) > 5:
        delta_thetas = [0.0]
        for i in range(1, len(vector_list)):
            x1, z1 = vector_list[i-1]
            x2, z2 = vector_list[i]
            # Normalizacija i skalarni/vektorski proizvod u 2D
            dot = x1*x2 + z1*z2
            det = x1*z2 - z1*x2
            norm = np.sqrt(x1**2 + z1**2) * np.sqrt(x2**2 + z2**2)
            if norm > 1e-6:
                cos_val = np.clip(dot / norm, -1.0, 1.0)
                d_th = np.arccos(cos_val)
                if det < 0: d_th = -d_th # Smer
            else:
                d_th = 0.0
            
            # Odbacivanje nerealnih skokova kamere
            if abs(d_th) > 0.5: d_th = 0.0
            delta_thetas.append(d_th)
            
        # Kumulativni ugao pa diferenciranje
        cumulative_theta = np.cumsum(delta_thetas)
        theta_smooth = savgol_filter(cumulative_theta, window_length=11, polyorder=2)
        omega_raw = np.gradient(theta_smooth, dt)
        
        # Uzimamo apsolutnu vrednost jer je smer rotacije uvek pozitivan u celom testu,
        # ili filtriramo da nema neželjenih obrnutih minusnih ispada
        omega_deg = np.abs(np.degrees(savgol_filter(omega_raw, window_length=7, polyorder=2)))
        
        alpha_raw = np.gradient(omega_raw, dt)
        alpha_smooth = savgol_filter(alpha_raw, window_length=9, polyorder=2)

    arr_inertia = np.array(inertia_list, dtype=float)
    arr_time = np.array(time_list, dtype=float)
    valid_mask = np.isfinite(arr_inertia)
    arr_inertia = arr_inertia[valid_mask]
    arr_time = arr_time[valid_mask]

    if len(arr_inertia) >= 11:
        inertia_smooth = savgol_filter(arr_inertia, window_length=11, polyorder=2)
    else:
        inertia_smooth = arr_inertia
        
    physics_results_dict[clean_name] = {
        "time": arr_time[:len(inertia_smooth)],
        "inertia": inertia_smooth,
        "omega": omega_deg[:len(inertia_smooth)],
        "alpha": alpha_smooth[:len(inertia_smooth)]
    }

bordo_shades = ['#800000', '#a52a2a', '#8b0000', '#b22222', '#cd5c5c', '#5c0606']

# ---------------------------------------------------------
# GRAFIK 1: MOMENT INERCIJE
# ---------------------------------------------------------
plt.figure(figsize=(12, 7), facecolor='#fff5f5')
ax = plt.gca()
ax.set_facecolor('#ffffff')

for i, (name, data) in enumerate(physics_results_dict.items()):
    plt.plot(data["time"], data["inertia"], linewidth=2.2, color=bordo_shades[i % len(bordo_shades)], label=name)

plt.title("Zavisnost momenta inercije (I) od vremena (t) tokom 4 sekunde rotacije", 
          fontsize=13, fontweight='bold', fontstyle='italic', color='#5c0606', pad=15, family='sans-serif')
plt.xlabel("Vreme [s]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.ylabel("Moment inercije I [kg·m²]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.legend(loc='upper right', frameon=True, facecolor='#fff0f0', edgecolor='#800000')
plt.grid(True, linestyle='--', alpha=0.5, color='#d4a373')
plt.xlim(0, 4.0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "grafikmomentinercije.png"), dpi=300, facecolor=plt.gcf().get_facecolor())
plt.close()
print("=== Sačuvan 1: grafikmomentinercije.png")

# ---------------------------------------------------------
# GRAFIK 2: UGAONA BRZINA (Sada ispravno u plusu!)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7), facecolor='#fff5f5')
ax = plt.gca()
ax.set_facecolor('#ffffff')

for i, (name, data) in enumerate(physics_results_dict.items()):
    plt.plot(data["time"], data["omega"], linewidth=2.2, color=bordo_shades[i % len(bordo_shades)], label=name)

plt.title(r"Zavisnost ugaone brzine ($\omega$) od vremena (t) tokom 4 sekunde rotacije", 
          fontsize=13, fontweight='bold', fontstyle='italic', color='#5c0606', pad=15, family='sans-serif')
plt.xlabel("Vreme [s]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.ylabel(r"Ugaona brzina $\omega$ [°/s]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.legend(loc='upper right', frameon=True, facecolor='#fff0f0', edgecolor='#800000')
plt.grid(True, linestyle='--', alpha=0.5, color='#d4a373')
plt.xlim(0, 4.0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "grafikugaonabrzina.png"), dpi=300, facecolor=plt.gcf().get_facecolor())
plt.close()
print("=== Sačuvan 2: grafikugaonabrzina.png")

# ---------------------------------------------------------
# GRAFIK 3: UGAONO UBRZANJE
# ---------------------------------------------------------
plt.figure(figsize=(12, 7), facecolor='#fff5f5')
ax = plt.gca()
ax.set_facecolor('#ffffff')

for i, (name, data) in enumerate(physics_results_dict.items()):
    plt.plot(data["time"], data["alpha"], linewidth=2.2, color=bordo_shades[i % len(bordo_shades)], label=name)

plt.title(r"Zavisnost ugaonog ubrzanja ($\alpha$) od vremena (t) tokom 4 sekunde rotacije", 
          fontsize=13, fontweight='bold', fontstyle='italic', color='#5c0606', pad=15, family='sans-serif')
plt.xlabel("Vreme [s]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.ylabel(r"Ugaono ubrzanje $\alpha$ [rad/s²]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.legend(loc='upper right', frameon=True, facecolor='#fff0f0', edgecolor='#800000')
plt.grid(True, linestyle='--', alpha=0.5, color='#d4a373')
plt.xlim(0, 4.0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "grafikugaonoubrzanje.png"), dpi=300, facecolor=plt.gcf().get_facecolor())
plt.close()
print("=== Sačuvan 3: grafikugaonoubrzanje.png")

print("\n=gotojoo: Sva 3 grafika su uspešno ispravljena i sačuvana u bordo stilu!\n")