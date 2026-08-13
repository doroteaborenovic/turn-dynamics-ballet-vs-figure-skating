import pandas as pd
import numpy as np
import cv2
import os
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=4.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    ser = pd.Series(data)
    filled = ser.interpolate(method='pchip', limit_direction='both').ffill().bfill().values
    if np.isnan(filled).any(): return data
    return filtfilt(b, a, filled)

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12)
]

input_dir = "konacne_koordinate"
output_dir = "konacni_videi"
os.makedirs(output_dir, exist_ok=True)

for csv_file in os.listdir(input_dir):
    if not csv_file.endswith("_konacne.csv"): continue
    
    athlete_key = csv_file.replace("_konacne.csv", "")
    csv_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    fps = df['FPS'].iloc[0]
    
    # Filtriranje mikro-drhtaja
    for lm_id in df['Landmark_ID'].unique():
        lm_mask = df['Landmark_ID'] == lm_id
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = df.loc[lm_mask, col].values
            df.loc[lm_mask, col] = butter_lowpass_filter(vals, cutoff=4.0, fs=fps)

    width, height = 1280, 720
    video_path = os.path.join(output_dir, f"video_{athlete_key}_konacno.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Fiksirana skala
    valid_x, valid_y = df['X_clean'].dropna(), df['Y_clean'].dropna()
    x_min, x_max = np.percentile(valid_x, 2), np.percentile(valid_x, 98)
    y_min, y_max = np.percentile(valid_y, 2), np.percentile(valid_y, 98)
    
    scale_x = (width * 0.5) / (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    scale_y = (height * 0.5) / (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    for frame_idx in sorted(df['Frame'].unique()):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frame_data = df[df['Frame'] == frame_idx]
        curr_time = frame_data['Time'].iloc[0]
        
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
                cv2.line(canvas, points_2d[p1], points_2d[p2], (0, 255, 100), 2, cv2.LINE_AA)
                
        cv2.putText(canvas, f"Atleta: {athlete_key.upper()} | Time: {curr_time:.2f}s | Frame: {frame_idx}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(canvas)
        
    out.release()
    print(f" Sačuvan konačni video: {video_path}")

print("=== RENDEROVANJE KONAČNIH PREGLEDA JE GOTOVO ===")
