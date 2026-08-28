import pandas as pd
import numpy as np
import cv2
import os
from scipy.signal import butter, filtfilt

# --- 1. BUTTERWORTH LOWPASS FILTER (Nivo 4, Cutoff 5Hz - Biomehanički standard) ---
def butter_lowpass_filter(data, cutoff=5.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    ser = pd.Series(data)
    # Pametna PCHIP interpolacija pre filtriranja da se izbegnu rupe
    filled = ser.interpolate(method='pchip', limit_direction='both').ffill().bfill().values
    
    if np.isnan(filled).any():
        return data
        
    y = filtfilt(b, a, filled)
    return y

# --- 2. SPOJEVI SKELETA ---
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Trup
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Leva ruka
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Desna ruka
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Leva noga
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # Desna noga
    (0, 11), (0, 12) # Glava
]

input_dir = "obradjene_koordinate"
original_videos_dir = r"C:\Users\PC\Videos\Screen Recordings\videiples"
output_dir = "renderovani_videi"
os.makedirs(output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

print("=== START STABILIZACIJE I RENDEROVANJA SKELETA NA TAČAN FPS ===")

for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    athlete_key = csv_file.replace("koordinate_", "").replace("_naucan.csv", "")
    
    df = pd.read_csv(csv_path)
    
    # Pronalaženje originalnog videa radi čitanja TAČNOG FPS-a
    orig_fps = 30.0
    for vid_name in os.listdir(original_videos_dir):
        if athlete_key in vid_name.lower() or vid_name.lower().startswith(athlete_key):
            orig_vid_path = os.path.join(original_videos_dir, vid_name)
            cap = cv2.VideoCapture(orig_vid_path)
            if cap.isOpened():
                orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            break
            
    print(f"Obrada: {athlete_key} | Prepoznat Tačan FPS: {orig_fps:.2f}")

    # --- STABILIZACIJA I GLATKA INTERPOLACIJA ZGLOBOVA ---
    for lm_id in df['Landmark_ID'].unique():
        lm_mask = df['Landmark_ID'] == lm_id
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = df.loc[lm_mask, col].values
            smooth_vals = butter_lowpass_filter(vals, cutoff=5.0, fs=orig_fps)
            df.loc[lm_mask, col] = smooth_vals

    # Dimenzije ekrana
    width, height = 1280, 720
    video_name = f"video_{athlete_key}_perfektan.mp4"
    video_path = os.path.join(output_dir, video_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, orig_fps, (width, height))
    
    # KORISTIMO GLOBILNU (FIKSIRANU) SKALU DA KOSTUR NE SKAČE I NE MENJA VELIČINU!
    valid_x = df['X_clean'].dropna()
    valid_y = df['Y_clean'].dropna()
    
    # Percentili 1% i 99% izbacuju ekstremne greške pri skaliranju
    x_min, x_max = np.percentile(valid_x, 1), np.percentile(valid_x, 99)
    y_min, y_max = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
    
    scale_x = (width * 0.55) / (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    scale_y = (height * 0.55) / (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    frames = sorted(df['Frame'].unique())

    for frame_idx in frames:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frame_data = df[df['Frame'] == frame_idx]
        
        points_2d = {}
        for _, row in frame_data.iterrows():
            lm_id = int(row['Landmark_ID'])
            x, y = row['X_clean'], row['Y_clean']
            
            if not np.isnan(x) and not np.isnan(y):
                px = int(width / 2 + (x - center_x) * scale)
                py = int(height / 2 - (y - center_y) * scale)
                points_2d[lm_id] = (px, py)
                cv2.circle(canvas, (px, py), 4, (255, 220, 0), -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in points_2d and p2 in points_2d:
                pt1, pt2 = points_2d[p1], points_2d[p2]
                cv2.line(canvas, pt1, pt2, (0, 255, 100), 2, cv2.LINE_AA)
                
        cv2.putText(canvas, f"Atleticar: {athlete_key} | FPS: {orig_fps:.2f} | Frame: {frame_idx}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(canvas)
        
    out.release()
    print(f" Uspešno kreiran video 100% sinhronizovane brzine: {video_path}\n")

print("=== SVI VIDEI SU ISPRAVLJENI I REPRODUKUJU SE U REALNOJ BRZINI! ===")