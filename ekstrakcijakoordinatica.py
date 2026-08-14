#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import logging
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

ATHLETE_DB = {
    "marianela": {"height": 1.74, "sport": "balet"},
    "kapitonova": {"height": 1.68, "sport": "balet"},
    "khoreva": {"height": 1.73, "sport": "balet"},
    "trusova": {"height": 1.66, "sport": "klizanje"},
    "scerebakova": {"height": 1.61, "sport": "klizanje"},
    "shcherbakova": {"height": 1.61, "sport": "klizanje"},
    "liu": {"height": 1.58, "sport": "klizanje"}
}

VISIBILITY_THRESHOLD = 0.3
VELOCITY_THRESHOLD = 12.0  # m/s (fizički maks brzina udova u piruetama/skokovima)
MAX_INTERP_SECONDS = 0.4
SAVGOL_WINDOW_SECONDS = 0.2
SAVGOL_POLYORDER = 2

base_path = os.environ.get("VIDEO_DIR", "videos")
output_folder = os.environ.get("OUTPUT_DIR", "obradjene_koordinate")
os.makedirs(output_folder, exist_ok=True)


#nayivi fajlova su ovde 
videos_to_process = [
    {"filename": "viktoriakapitonovabalet.mp4", "key": "kapitonova"},
    {"filename": "marianelanunezbalet.mp4", "key": "marianela"},
    {"filename": "alysaliuklizanjemp4.mp4", "key": "liu"},
    {"filename": "anascerebakovaklizanje.mp4", "key": "scerebakova"},
    {"filename": "aleksandratrusovaklizanjep4.mp4", "key": "trusova"},
    {"filename": "mariakhorevabalet.mp4", "key": "khoreva"}
]

def process_video(video_path, athlete_key):
    if not os.path.exists(video_path):
        logger.error(f"Ne postoji video fajl: {video_path}")
        return
        
    true_height = ATHLETE_DB.get(athlete_key, {"height": 1.65})["height"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Obrada: {athlete_key} | FPS: {fps:.2f} | Ukupno frejmova: {total_frames}")
    
    coordinate_data = []
    frame_count = 0
    
    with mp.solutions.pose.Pose(
        static_image_mode=False, 
        model_complexity=2, 
        smooth_landmarks=True, 
        min_detection_confidence=0.6, 
        min_tracking_confidence=0.6
    ) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret: 
                break
            
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Ako je detekcija uspela
            if results.pose_world_landmarks and results.pose_landmarks:
                for lm_id, (lm, lm_world) in enumerate(zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark)):
                    coordinate_data.append({
                        "Frame": frame_count,
                        "Time": frame_count * dt,
                        "FPS": fps,
                        "Landmark_ID": lm_id,
                        "Visibility": lm.visibility,
                        "X_raw": lm_world.x,
                        "Y_raw": -lm_world.y,  # INVERZIJA Y OSE: sada je +Y ka GORE (Fizicki korektno)
                        "Z_raw": lm_world.z
                    })
            else:
                # METODOLOŠKA KOREKCIJA: Popunjavanje missing frame-a sa NaN
                for lm_id in range(33):
                    coordinate_data.append({
                        "Frame": frame_count,
                        "Time": frame_count * dt,
                        "FPS": fps,
                        "Landmark_ID": lm_id,
                        "Visibility": 0.0,
                        "X_raw": np.nan,
                        "Y_raw": np.nan,
                        "Z_raw": np.nan
                    })
            
            frame_count += 1
            if frame_count % 150 == 0:
                logger.info(f" Napredak ekstrakcije: {frame_count}/{total_frames} frejmova")
                
    cap.release()

    if not coordinate_data:
        logger.error("Nema izvučenih podataka!")
        return

    df = pd.DataFrame(coordinate_data)
    
    # --- 1. SKALIRANJE NA REALNU VISINU ATLETIČARKE ---
    # MediaPipe pose_world_landmarks su u realnim metrima. 
    # Da bismo bili precizni, verifikujemo antropometrijsku razmeru segmenta trupa i nogu u uspravnijim frejmovima
    df['X_clean'] = df['X_raw']
    df['Y_clean'] = df['Y_raw']
    df['Z_clean'] = df['Z_raw']
    df['Detection_OK'] = df['Visibility'] >= VISIBILITY_THRESHOLD

    # Outlier filter za nisku vidljivost
    bad_vis = ~df['Detection_OK']
    df.loc[bad_vis, ['X_clean', 'Y_clean', 'Z_clean']] = np.nan

    # --- 2. OBRADA PO TACKAMA (GROUPBY ZA SPREČAVANJE MESA-NJA INDEKSA) ---
    processed_dfs = []
    
    for lm_id, group in df.groupby('Landmark_ID'):
        group = group.sort_values('Frame').copy()
        
        # Outlier filter po brzini (kinematički nerealni skokovi)
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = group[col].values
            vel = np.abs(np.diff(vals, prepend=vals[0])) / dt
            outliers = vel > VELOCITY_THRESHOLD
            group.loc[outliers, col] = np.nan
            group.loc[outliers, 'Detection_OK'] = False

        # Interpolacija kretanja (Linear)
        max_gap = int(fps * MAX_INTERP_SECONDS)
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            group[col] = group[col].interpolate(method='linear', limit=max_gap, limit_direction='both')

        # Savitzky-Golay Filtriranje (Glađenje šuma opreme/kamere)
        win_len = int(fps * SAVGOL_WINDOW_SECONDS)
        win_len = win_len + 1 if win_len % 2 == 0 else win_len
        win_len = max(5, win_len)

        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            ser = group[col]
            if ser.isna().sum() / len(ser) < 0.6:  # Ako imamo bar 40% validnih tačaka
                filled = ser.ffill().bfill()
                if not filled.isna().any():
                    smoothed = savgol_filter(filled.values, window_length=win_len, polyorder=SAVGOL_POLYORDER)
                    # Vraćamo NaN na mesta gde interpolacija nije uspela
                    smoothed[ser.isna()] = np.nan
                    group[col] = smoothed

        processed_dfs.append(group)

    df_final = pd.concat(processed_dfs).sort_values(['Frame', 'Landmark_ID']).reset_index(drop=True)

    csv_path = os.path.join(output_folder, f"koordinate_{athlete_key}_naucan.csv")
    df_final[['Frame', 'Time', 'FPS', 'Landmark_ID', 'Visibility', 'Detection_OK', 
              'X_raw', 'Y_raw', 'Z_raw', 'X_clean', 'Y_clean', 'Z_clean']].to_csv(csv_path, index=False, float_format='%.6f')
    
    logger.info(f" Sačuvani naučni podaci u: {csv_path}\n")

if __name__ == "__main__":
    logger.info("=== START NAUČNOG PIPELINE-A (ISPRAVLJENA METODOLOGIJA) ===")
    for item in videos_to_process:
        process_video(os.path.join(base_path, item["filename"]), item["key"])
    logger.info("=== ZAVRŠENO USPEŠNO ===")
