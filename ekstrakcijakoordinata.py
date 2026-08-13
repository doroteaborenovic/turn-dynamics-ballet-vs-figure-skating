import pandas as pd
import numpy as np
import cv2
import os
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import PchipInterpolator

EXACT_SECONDS = {
    "marianela": (2.0, 6.0),
    "kapitonova": (0.0, 4.0),
    "khoreva": (1.0, 5.0),
    "liu": (1.0, 5.0),
    "scerebakova": (5.0, 9.0),
    "trusova": (6.0, 10.0)
}

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

# --- 1. DETEKCIJA POKVARENIH FREJMOVA PREKO 3D ANATOMSKE DISTANCE KOSTIJU ---
def detect_anatomical_deformations(df, frames):
    bad_frames = set()
    
    # Računamo 3D širinu ramena (11-12) i kukova (23-24) za svaki frejm
    shoulder_dists = []
    hip_dists = []
    
    for f in frames:
        f_df = df[df['Frame'] == f]
        if f_df.empty: continue
        
        # Ramena 11 i 12
        p11 = f_df[f_df['Landmark_ID'] == 11][['X_clean', 'Y_clean', 'Z_clean']].values
        p12 = f_df[f_df['Landmark_ID'] == 12][['X_clean', 'Y_clean', 'Z_clean']].values
        
        if len(p11) > 0 and len(p12) > 0 and not np.isnan(p11).any() and not np.isnan(p12).any():
            d_sh = np.linalg.norm(p11 - p12)
            shoulder_dists.append((f, d_sh))
            
        # Kukovi 23 i 24
        p23 = f_df[f_df['Landmark_ID'] == 23][['X_clean', 'Y_clean', 'Z_clean']].values
        p24 = f_df[f_df['Landmark_ID'] == 24][['X_clean', 'Y_clean', 'Z_clean']].values
        
        if len(p23) > 0 and len(p24) > 0 and not np.isnan(p23).any() and not np.isnan(p24).any():
            d_hip = np.linalg.norm(p23 - p24)
            hip_dists.append((f, d_hip))

    if not shoulder_dists: return bad_frames
    
    sh_df = pd.DataFrame(shoulder_dists, columns=['Frame', 'Dist'])
    med_sh = sh_df['Dist'].median()
    
    # detekcija pokavrenog frejma ako se ispuni ovaj ulsov 
    bad_sh = sh_df[(sh_df['Dist'] < 0.4 * med_sh) | (sh_df['Dist'] > 1.8 * med_sh)]['Frame'].values
    bad_frames.update(bad_sh)
    
    if hip_dists:
        hip_df = pd.DataFrame(hip_dists, columns=['Frame', 'Dist'])
        med_hip = hip_df['Dist'].median()
        bad_hip = hip_df[(hip_df['Dist'] < 0.4 * med_hip) | (hip_df['Dist'] > 1.8 * med_hip)]['Frame'].values
        bad_frames.update(bad_hip)
        
    return bad_frames

# --- 2. ADVANCED GLAĐENJE SA PCHIP INTERPOLACIJOM I BUTTERWORTH FILTEROM ---
def smooth_landmark_series(series, fps=30.0):
    vals = series.copy().values.astype(float)
    if len(vals) < 5: return vals
    
    # a) Median filter eliminise mikro-spikove
    vals = medfilt(vals, kernel_size=3)
    
    # b) Rolling median outlier removal
    ser = pd.Series(vals)
    rolling_med = ser.rolling(window=5, center=True, min_periods=1).median()
    diff = np.abs(ser - rolling_med)
    ser[diff > 0.12] = np.nan
    
    # c) PCHIP Hermite interpolacija (savršeno spaja prirodne krive)
    valid_mask = ~ser.isna()
    if valid_mask.sum() >= 4:
        try:
            x_val = np.where(valid_mask)[0]
            y_val = ser[valid_mask].values
            pchip = PchipInterpolator(x_val, y_val, extrapolate=True)
            filled = pchip(np.arange(len(ser)))
        except Exception:
            filled = ser.interpolate(method='linear', limit_direction='both').ffill().bfill().values
    else:
        filled = ser.ffill().bfill().values
        
    # d) Butterworth Lowpass Filter (3.5 Hz)
    nyq = 0.5 * fps
    b, a = butter(3, min(3.5 / nyq, 0.95), btype='low', analog=False)
    smoothed = filtfilt(b, a, filled, padlen=min(10, len(filled)-1))
    
    return smoothed

print("=== START ANATOMSKE KOREKCIJE I PEGLANJA KOORDINATA ===")

for csv_file in os.listdir(input_dir):
    if not csv_file.endswith(".csv"): continue
    
    athlete_key = csv_file.replace("koordinate_", "").replace("_naucan.csv", "")
    csv_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    # Čitanje tačnog FPS-a
    orig_fps = 30.0
    for vid_name in os.listdir(original_videos_dir):
        if athlete_key in vid_name.lower() or vid_name.lower().startswith(athlete_key):
            orig_vid_path = os.path.join(original_videos_dir, vid_name)
            cap = cv2.VideoCapture(orig_vid_path)
            if cap.isOpened():
                orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            break
            
    dt = 1.0 / orig_fps
    
    start_sec, end_sec = EXACT_SECONDS.get(athlete_key, (0.0, 4.0))
    start_f = int(start_sec * orig_fps)
    end_f = start_f + int(4.0 * orig_fps)
    
    df_cropped = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    frames = sorted(df_cropped['Frame'].unique())
    
    # 1. Detekcija izdeformisanih frejmova
    bad_frames = detect_anatomical_deformations(df_cropped, frames)
    print(f"\nAtleta: {athlete_key.upper()} | Detektovano izdeformisanih frejmova: {len(bad_frames)}")
    
    # 2. Pretvaramo pokvarene frejmove u NaN
    if bad_frames:
        for f in bad_frames:
            df_cropped.loc[df_cropped['Frame'] == f, ['X_clean', 'Y_clean', 'Z_clean']] = np.nan
            
    # 3. PCHIP Interpolacija + Butterworth filter po svakom landmark-u
    for lm_id in df_cropped['Landmark_ID'].unique():
        lm_mask = df_cropped['Landmark_ID'] == lm_id
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            df_cropped.loc[lm_mask, col] = smooth_landmark_series(df_cropped.loc[lm_mask, col], fps=orig_fps)

    # Standardizacija vremenske ose (0.00s do 4.00s)
    df_cropped['Original_Frame'] = df_cropped['Frame']
    df_cropped['Frame'] = df_cropped['Frame'] - start_f
    df_cropped['Time'] = df_cropped['Frame'] * dt
    df_cropped['FPS'] = orig_fps
    
    # Export CSV
    final_csv_name = f"{athlete_key}_konacne.csv"
    final_csv_path = os.path.join(output_csv_dir, final_csv_name)
    cols_order = ['Frame', 'Time', 'FPS', 'Original_Frame', 'Landmark_ID', 'Visibility', 'Detection_OK',
                  'X_clean', 'Y_clean', 'Z_clean', 'X_raw', 'Y_raw', 'Z_raw']
    df_cropped[cols_order].to_csv(final_csv_path, index=False, float_format='%.6f')
    
    # RENDEROVANJE ISPRAVLJENOG PREGLEDA
    width, height = 1280, 720
    video_path = os.path.join(output_vid_dir, f"video_{athlete_key}_konacno.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (width, height))
    
    valid_x, valid_y = df_cropped['X_clean'].dropna(), df_cropped['Y_clean'].dropna()
    x_min, x_max = np.percentile(valid_x, 2), np.percentile(valid_x, 98)
    y_min, y_max = np.percentile(valid_y, 2), np.percentile(valid_y, 98)
    
    scale = min((width * 0.5) / (x_max - x_min), (height * 0.5) / (y_max - y_min)) if (x_max - x_min) > 0 else 1.0
    center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    for f_idx in sorted(df_cropped['Frame'].unique()):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frame_data = df_cropped[df_cropped['Frame'] == f_idx]
        curr_t = frame_data['Time'].iloc[0]
        
        points_2d = {}
        for _, row in frame_data.iterrows():
            lm_id = int(row['Landmark_ID'])
            x, y = row['X_clean'], row['Y_clean']
            px = int(width / 2 + (x - center_x) * scale)
            py = int(height / 2 - (y - center_y) * scale)
            points_2d[lm_id] = (px, py)
            cv2.circle(canvas, (px, py), 4, (255, 220, 0), -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in points_2d and p2 in points_2d:
                cv2.line(canvas, points_2d[p1], points_2d[p2], (0, 255, 100), 2, cv2.LINE_AA)
                
        cv2.putText(canvas, f"Atleta: {athlete_key.upper()} | Time: {curr_t:.2f}s | Frame: {f_idx}/120", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(canvas)
        
    out.release()
    print(f" Sačuvan korigovan CSV i Video za: {athlete_key.upper()}")

print ("gotojooo")
