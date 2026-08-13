import pandas as pd
import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
#parametri i to on klasiak
#ovde ide racunanje momenta inercije i kao grafik zavisnsoti i od vremena od 4s 
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

input_coord_dir = "koordinate_sredjene"
if not os.path.exists(input_coord_dir):
    input_coord_dir = "obradjene_koordinate"

output_folder = "tabele_fizika"
os.makedirs(output_folder, exist_ok=True)

all_files = glob.glob(os.path.join(input_coord_dir, "*.csv"))

print(f"\nucitavanje koordinata iy foldera: {input_coord_dir}")
print(f"Pronađeno fajlova: {len(all_files)}\n")

inertia_data_dict = {}

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
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).replace("_sredjene.csv", "").replace(".csv", "").upper()

    # Određivanje pivot noge
    foot_left = df[df['Landmark_ID'] == 31][['X_clean', 'Y_clean', 'Z_clean']].values
    foot_right = df[df['Landmark_ID'] == 32][['X_clean', 'Y_clean', 'Z_clean']].values
    move_left = np.sum(np.std(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    move_right = np.sum(np.std(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    planted_foot_id = 31 if move_left <= move_right else 32

    frames_sorted = sorted(df['Frame'].unique())
    inertia_list = []
    time_list = []

    for frame_idx in frames_sorted:
        f_df = df[df['Frame'] == frame_idx].set_index('Landmark_ID')
        if f_df.empty: continue
        
        frame_pts = {lm_id: f_df.loc[lm_id, ['X_clean', 'Y_clean', 'Z_clean']].values.astype(float) for lm_id in f_df.index if lm_id in f_df.index}
        curr_time = f_df['Time'].iloc[0] if 'Time' in f_df.columns else frame_idx / 30.0
        
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

    # Čišćenje niza od bilo kakvih NaN ili Inf vrednosti pre filtriranja
    arr_inertia = np.array(inertia_list, dtype=float)
    arr_time = np.array(time_list, dtype=float)
    
    valid_mask = np.isfinite(arr_inertia)
    arr_inertia = arr_inertia[valid_mask]
    arr_time = arr_time[valid_mask]

    # Blago glađenje (savgol filter)
    if len(arr_inertia) > 10:
        window = 11 if len(arr_inertia) >= 11 else (len(arr_inertia) // 2) * 2 + 1
        if window >= 5:
            inertia_smooth = savgol_filter(arr_inertia, window_length=window, polyorder=2)
        else:
            inertia_smooth = arr_inertia
    else:
        inertia_smooth = arr_inertia
        
    inertia_data_dict[clean_name] = (arr_time[:len(inertia_smooth)], inertia_smooth)

# boje grafika i to
plt.figure(figsize=(12, 7), facecolor='#fff5f5')
ax = plt.gca()
ax.set_facecolor('#ffffff')

bordo_shades = ["#A50000", "#6aa52a", "#10008b", "#440027", "#000000", "#ffa200"]

for i, (name, (t_axis, i_vals)) in enumerate(inertia_data_dict.items()):
    if len(t_axis) > 0 and len(i_vals) > 0:
        color = bordo_shades[i % len(bordo_shades)]
        plt.plot(t_axis, i_vals, linewidth=2.2, color=color, label=name)

plt.title("Zavisnost momenta inercije (I) od vremena (t) tokom 4 sekunde rotacije", 
          fontsize=13, fontweight='bold', color='#5c0606', pad=15, family='sans-serif')
plt.xlabel("Vreme [s]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')
plt.ylabel("Moment inercije I [kg·m²]", fontsize=11, fontweight='bold', color='#5c0606', family='sans-serif')

plt.legend(loc='upper right', frameon=True, facecolor='#fff0f0', edgecolor='#800000')
plt.grid(True, linestyle='--', alpha=0.5, color='#d4a373')
plt.xlim(0, 4.0)

plt.tight_layout()
graph_path = os.path.join(output_folder, "grafikmomentinercije.png")
plt.savefig(graph_path, dpi=300, facecolor=plt.gcf().get_facecolor())
plt.close()

print(f"=gotojoo:\n{graph_path}\n")
