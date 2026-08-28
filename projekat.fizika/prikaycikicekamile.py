import pandas as pd
import numpy as np
import cv2
import os
from scipy.signal import butter, filtfilt

# --- TAČNI 4.0s INTERVALI ZA SVE ATLETIČARKE (Uključujući i Kamilu) ---
EXACT_SECONDS = {
    "marianela": (2.0, 6.0),
    "kapitonova": (0.0, 4.0),
    "khoreva": (1.0, 5.0),
    "liu": (1.0, 5.0),
    "scerebakova": (5.0, 9.0),
    "trusova": (6.0, 10.0),
    "kamilavalieva": (2.0, 6.0),
    "kamilavalievaklizanje": (2.0, 6.0),
    "kamila": (2.0, 6.0)
}

def clean_and_smooth_trajectory(data_series, fps=30.0):
    s = pd.Series(data_series).copy().astype(float)
    s[s.abs() > 1.5] = np.nan
    rolling_med = s.rolling(window=5, center=True, min_periods=1).median()
    diff = np.abs(s - rolling_med)
    s[diff > 0.15] = np.nan
    
    filled = s.interpolate(method='linear', limit_direction='both').ffill().bfill().values
    
    nyq = 0.5 * fps
    b, a = butter(4, min(4.0 / nyq, 0.99), btype='low', analog=False)
    smoothed = filtfilt(b, a, filled)
    
    return smoothed

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12)
]

input_dir = "obradjene_koordinate"
original_videos_dir = r"C:\Users\PC\Videos\Screen Recordings\videiples"
output_csv_dir = "konacne_koordinate"
output_vid_dir = "konacni_videi"

os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_vid_dir, exist_ok=True)

print("=== START UNIVERZALNE OBRADE (UKLJUČUJUĆI KAMILU) ===\n")

if not os.path.exists(input_dir):
    print(f"❌ GREŠKA: Ulazni folder '{input_dir}' ne postoji!")
else:
    for csv_file in os.listdir(input_dir):
        if not csv_file.endswith(".csv"): continue
        
        raw_key = csv_file.replace("koordinate_", "").replace("_naucan.csv", "").replace("_konacne.csv", "").replace("_sredjene", "").lower()
        
        athlete_key = None
        for k in EXACT_SECONDS.keys():
            if k in raw_key:
                athlete_key = k
                break
        if not athlete_key:
            athlete_key = raw_key

        csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(csv_path)
        
        # Čitanje tačnog FPS-a
        orig_fps = 30.0
        if os.path.exists(original_videos_dir):
            for vid_name in os.listdir(original_videos_dir):
                vid_lower = vid_name.lower()
                # Fleksibilnije prepoznavanje videa za svaku atlatu
                match = False
                if athlete_key in vid_lower:
                    match = True
                elif "kamila" in athlete_key and "kamila" in vid_lower:
                    match = True
                elif "valieva" in athlete_key and "valieva" in vid_lower:
                    match = True
                    
                if match:
                    orig_vid_path = os.path.join(original_videos_dir, vid_name)
                    cap = cv2.VideoCapture(orig_vid_path)
                    if cap.isOpened():
                        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        cap.release()
                    break
                
        dt = 1.0 / orig_fps
        
        start_sec, end_sec = EXACT_SECONDS.get(athlete_key, (2.0, 6.0))
        start_frame = int(start_sec * orig_fps)
        end_frame = start_frame + int(4.0 * orig_fps)
        
        if 'Frame' in df.columns:
            df_cropped = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)].copy()
        else:
            df_cropped = df.copy()
            
        if df_cropped.empty:
            df_cropped = df.copy()

        for lm_id in df_cropped['Landmark_ID'].unique():
            lm_mask = df_cropped['Landmark_ID'] == lm_id
            for col in ['X_clean', 'Y_clean', 'Z_clean']:
                if col in df_cropped.columns:
                    vals = df_cropped.loc[lm_mask, col]
                    df_cropped.loc[lm_mask, col] = clean_and_smooth_trajectory(vals, fps=orig_fps)

        if 'Frame' in df_cropped.columns:
            df_cropped['Original_Frame'] = df_cropped['Frame']
            df_cropped['Frame'] = df_cropped['Frame'] - df_cropped['Frame'].min()
            df_cropped['Time'] = df_cropped['Frame'] * dt
        df_cropped['FPS'] = orig_fps
        
        if "kamila" in athlete_key or "valieva" in athlete_key:
            clean_name = "kamilavalieva"
        else:
            clean_name = athlete_key
            
        final_csv_name = f"{clean_name}_konacne.csv"
        final_csv_path = os.path.join(output_csv_dir, final_csv_name)
        
        for col in ['Visibility', 'Detection_OK', 'X_raw', 'Y_raw', 'Z_raw']:
            if col not in df_cropped.columns:
                df_cropped[col] = 1.0 if 'Visibility' in col else 0.0
                
        cols_order = [c for c in ['Frame', 'Time', 'FPS', 'Original_Frame', 'Landmark_ID', 'Visibility', 'Detection_OK',
                      'X_clean', 'Y_clean', 'Z_clean', 'X_raw', 'Y_raw', 'Z_raw'] if c in df_cropped.columns]
        df_cropped[cols_order].to_csv(final_csv_path, index=False, float_format='%.6f')
        
        width, height = 1280, 720
        video_path = os.path.join(output_vid_dir, f"video_{clean_name}_konacno.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (width, height))
        
        valid_x = df_cropped['X_clean'].dropna() if 'X_clean' in df_cropped.columns else pd.Series([0])
        valid_y = df_cropped['Y_clean'].dropna() if 'Y_clean' in df_cropped.columns else pd.Series([0])
        if len(valid_x) > 0 and len(valid_y) > 0:
            x_min, x_max = np.percentile(valid_x, 1), np.percentile(valid_x, 99)
            y_min, y_max = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
            span_x = max(x_max - x_min, 0.5)
            span_y = max(y_max - y_min, 0.5)
            scale = min((width * 0.6) / span_x, (height * 0.6) / span_y)
        else:
            scale = 300.0

        for f_idx in sorted(df_cropped['Frame'].unique()):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            frame_data = df_cropped[df_cropped['Frame'] == f_idx]
            curr_t = frame_data['Time'].iloc[0] if 'Time' in frame_data.columns else f_idx / orig_fps
            
            trunk_pts = frame_data[frame_data['Landmark_ID'].isin([11, 12, 23, 24])] if 'Landmark_ID' in frame_data.columns else pd.DataFrame()
            center_x = trunk_pts['X_clean'].dropna().mean() if not trunk_pts.empty and 'X_clean' in trunk_pts.columns else 0.0
            center_y = trunk_pts['Y_clean'].dropna().mean() if not trunk_pts.empty and 'Y_clean' in trunk_pts.columns else 0.0
            if np.isnan(center_x) or np.isnan(center_y):
                center_x, center_y = 0.0, 0.0

            points_2d = {}
            for _, row in frame_data.iterrows():
                lm_id = int(row['Landmark_ID'])
                x, y = row['X_clean'], row['Y_clean']
                if np.isfinite(x) and np.isfinite(y):
                    px = int(width / 2 + (x - center_x) * scale)
                    py = int(height / 2 - (y - center_y) * scale)
                    points_2d[lm_id] = (px, py)
                    cv2.circle(canvas, (px, py), 4, (255, 220, 0), -1)

            for p1, p2 in POSE_CONNECTIONS:
                if p1 in points_2d and p2 in points_2d:
                    cv2.line(canvas, points_2d[p1], points_2d[p2], (0, 255, 100), 2, cv2.LINE_AA)
                    
            cv2.putText(canvas, f"Atleta: {clean_name.upper()} | Time: {curr_t:.2f}s", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(canvas)
            
        out.release()
        print(f"✅ Obrađen i sačuvan CSV/Video za: {clean_name.upper()}")

print("\n=== SVE JE USPEŠNO OBRADJENO I SAČUVANO U 'konacne_koordinate' I 'konacni_videi'! ===")