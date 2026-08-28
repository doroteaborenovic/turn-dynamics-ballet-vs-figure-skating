import pandas as pd
import numpy as np
import cv2
import os
import mediapipe as mp

# ============================================================
# KONFIGURACIJA
# ============================================================
VIDEO_PATH = r"C:\Users\PC\Videos\Screen Recordings\videiples\kamilavalievaklizanje.mp4"
ATHLETE_KEY = "kamilavalieva"

OUTPUT_CSV_DIR = "obradjene_koordinate"
OUTPUT_VID_DIR = "renderovani_videi"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_VID_DIR, exist_ok=True)

# --- SPOJEVI SKELETA (MediaPipe standard) ---
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Trup
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Leva ruka
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Desna ruka
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Leva noga
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # Desna noga
    (0, 11), (0, 12) # Glava
]

def main():
    print("=== START EKSTRAKCIJE I RENDEROVANJA SKELETA ===")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ GREŠKA: Video nije pronađen na putanji:\n  {VIDEO_PATH}")
        return

    # --------------------------------------------------------
    # KORAK 1: Direktna ekstrakcija 33 tačke preko MediaPipe-a
    # --------------------------------------------------------
    print(f"\n[1/2] Učitavam video i vršim MediaPipe detekciju 33 tačke...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    data_rows = []
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            curr_time = frame_idx / orig_fps
            
            if results.pose_world_landmarks and results.pose_landmarks:
                world_landmarks = results.pose_world_landmarks.landmark
                image_landmarks = results.pose_landmarks.landmark
                
                for lm_id in range(33):
                    img_lm = image_landmarks[lm_id]
                    world_lm = world_landmarks[lm_id]
                    
                    data_rows.append({
                        'Frame': frame_idx,
                        'Time': curr_time,
                        'FPS': orig_fps,
                        'Landmark_ID': lm_id,
                        'Visibility': img_lm.visibility,
                        'X_clean': img_lm.x * width_orig,
                        'Y_clean': (1.0 - img_lm.y) * height_orig, # Inverzija Y ose da bude uspravno
                        'Z_clean': img_lm.z * width_orig,
                        'X_raw': world_lm.x,
                        'Y_raw': world_lm.y,
                        'Z_raw': world_lm.z
                    })
            else:
                # Ako u frejmu nema detekcije, upisujemo NaN da ne pukne
                for lm_id in range(33):
                    data_rows.append({
                        'Frame': frame_idx,
                        'Time': curr_time,
                        'FPS': orig_fps,
                        'Landmark_ID': lm_id,
                        'Visibility': 0.0,
                        'X_clean': np.nan,
                        'Y_clean': np.nan,
                        'Z_clean': np.nan,
                        'X_raw': np.nan,
                        'Y_raw': np.nan,
                        'Z_raw': np.nan
                    })
            
            if frame_idx % 30 == 0:
                print(f"  Obrađeno frejmova: {frame_idx}/{total_frames}", end='\r')
            frame_idx += 1

    cap.release()
    df = pd.DataFrame(data_rows)
    
    csv_path = os.path.join(OUTPUT_CSV_DIR, f"koordinate_{ATHLETE_KEY}_naucan.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Koordinate sačuvane u: {csv_path}")

    # --------------------------------------------------------
    # KORAK 2: Renderovanje tačno po tvojoj logici
    # --------------------------------------------------------
    print(f"\n[2/2] Renderujem video za: {ATHLETE_KEY} | FPS: {orig_fps:.2f}")
    
    width, height = 1280, 720
    video_name = f"video_{ATHLETE_KEY}_perfektan.mp4"
    video_path = os.path.join(OUTPUT_VID_DIR, video_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, orig_fps, (width, height))
    
    valid_x = df['X_clean'].dropna()
    valid_y = df['Y_clean'].dropna()
    
    if len(valid_x) > 0 and len(valid_y) > 0:
        x_min, x_max = np.percentile(valid_x, 1), np.percentile(valid_x, 99)
        y_min, y_max = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
    else:
        x_min, x_max = 0, width_orig
        y_min, y_max = 0, height_orig
    
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
                
        cv2.putText(canvas, f"Atleticar: {ATHLETE_KEY} | FPS: {orig_fps:.2f} | Frame: {frame_idx}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(canvas)
        
    out.release()
    print(f" Uspešno kreiran video: {video_path}\n")
    print("=== KRAJ OBRADE ===")

if __name__ == "__main__":
    main()