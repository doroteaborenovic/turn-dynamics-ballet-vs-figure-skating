import pandas as pd
import numpy as np
import cv2
import os
from scipy.signal import butter, filtfilt

# --- TAČNI 4.0s INTERVALI ZA SVIH 6 ATLETIČARKI ---
EXACT_SECONDS = {
    "marianela": (2.0, 6.0),
    "kapitonova": (0.0, 4.0),
    "khoreva": (1.0, 5.0),
    "liu": (1.0, 5.0),
    "scerebakova": (5.0, 9.0),
    "trusova": (6.0, 10.0)
}

# JEDNOSTAVNO I SIGURNO PEGLANJE KOORDINATA (BEZ EKSTRAPOLACIJE I LINIJA PO ĆOŠKOVIMA)
def clean_simple_series(series, fps=30.0):
    s = pd.Series(series).copy().astype(float)
    
    # 1. Ako jak skok ode van realnog okvira tela (preko 2.5 metra), stavi NaN
    s[s.abs() > 2.5] = np.nan
    
    # 2. Linearna interpolacija unutar granica (GARANTOVANO NE MOŽE ODLETETI VAN EKRANA)
    s = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    # 3. Butterworth Lowpass filter (4.5 Hz) za lep i stabilan pokret
    nyq = 0.5 * fps
    b, a = butter(3, min(4.5 / nyq, 0.95), btype='low', analog=False)
    
    if len(s) > 10:
        smoothed = filtfilt(b, a, s.values)
    else:
        smoothed = s.values
        
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

print("=== START SIGURNOG I JEDNOSTAVNOG PEGLANJA KOORDINATA ===\n")

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
    
    # Tačno 4.00s (121 frejm)
    start_sec, end_sec = EXACT_SECONDS.get(athlete_key, (0.0, 4.0))
    start_frame = int(start_sec * orig_fps)
    end_frame = start_frame + int(4.0 * orig_fps)
    
    df_cropped = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)].copy()
    
    # Čišćenje svake koordinate bez ikakvih ekstrapolacija
    for lm_id in df_cropped['Landmark_ID'].unique():
        lm_mask = df_cropped['Landmark_ID'] == lm_id
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = df_cropped.loc[lm_mask, col].values
            df_cropped.loc[lm_mask, col] = clean_simple_series(vals, fps=orig_fps)

    # Standardizacija vremenske ose (0.00s do 4.00s)
    df_cropped['Original_Frame'] = df_cropped['Frame']
    df_cropped['Frame'] = df_cropped['Frame'] - start_frame
    df_cropped['Time'] = df_cropped['Frame'] * dt
    df_cropped['FPS'] = orig_fps
    
    # Sačuvanje čistog CSV-a
    final_csv_name = f"{athlete_key}_konacne.csv"
    final_csv_path = os.path.join(output_csv_dir, final_csv_name)
    cols_order = ['Frame', 'Time', 'FPS', 'Original_Frame', 'Landmark_ID', 'Visibility', 'Detection_OK',
                  'X_clean', 'Y_clean', 'Z_clean', 'X_raw', 'Y_raw', 'Z_raw']
    df_cropped[cols_order].to_csv(final_csv_path, index=False, float_format='%.6f')
    
    # RENDEROVANJE NORMALNOG VIDEO PREGLEDA
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
            
            # Crtamo samo ako je tačka unutar ekrana
            if 0 <= px < width and 0 <= py < height:
                points_2d[lm_id] = (px, py)
                cv2.circle(canvas, (px, py), 4, (255, 220, 0), -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in points_2d and p2 in points_2d:
                cv2.line(canvas, points_2d[p1], points_2d[p2], (0, 255, 100), 2, cv2.LINE_AA)
                
        cv2.putText(canvas, f"Atleta: {athlete_key.upper()} | Time: {curr_t:.2f}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(canvas)
        
    out.release()
    print(f" Sačuvan NORMALAN i čist CSV/Video za: {athlete_key.upper()}")

print("\n=== SVE LINIJE SU UKLONJENE, PODACI SU CISTI I NORMALNI! ===")
