import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
from scipy.signal import savgol_filter

# ---------------------------------------------------------
# BAZA SPORTISTA (Za kalibraciju pravih metara)
# ---------------------------------------------------------
ATHLETE_DB = {
    "marianela": {"height": 1.74},
    "kapitonova": {"height": 1.68},
    "khoreva": {"height": 1.73},
    "trusova": {"height": 1.66},
    "scerebakova": {"height": 1.61},
    "shcherbakova": {"height": 1.61},
    "liu": {"height": 1.58}
}

mp_pose = mp.solutions.pose

base_path = r"C:\Users\PC\Videos\Screen Recordings"
output_folder = os.path.join(base_path, "processed_coordinates_scientific")
os.makedirs(output_folder, exist_ok=True)

videos_to_process = [
    {"filename": "viktoriakapitonovabalet.mp4", "key": "kapitonova"},
    {"filename": "marianelanunezbalet.mp4", "key": "marianela"},
    {"filename": "alysaliuklizanjemp4.mp4", "key": "liu"},
    {"filename": "anascerebakovaklizanje.mp4", "key": "scerebakova"},
    {"filename": "aleksandratrusovaklizanjep4.mp4", "key": "trusova"},
    {"filename": "mariakhorevabalet.mp4", "key": "khoreva"}
]

print("="*85)
print(" POKREĆEM NAUČNI PIPELINE: RAW -> OUTLIER REMOVAL -> ADAPTIVE SMOOTHING -> CSV")
print("="*85)

for index, item in enumerate(videos_to_process, 1):
    video_filename = item["filename"]
    athlete_key = item["key"]
    full_video_path = os.path.join(base_path, video_filename)
    
    if not os.path.exists(full_video_path):
        continue
        
    print(f"\n[{index}/6] Ekstrakcija i obrada: '{athlete_key}'...")
    cap = cv2.VideoCapture(full_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    dt = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dinamički prozor za SavGol filter (približno 0.25 sekundi pokreta)
    # Mora biti neparan broj
    window_length = int(fps * 0.25)
    if window_length % 2 == 0: window_length += 1
    if window_length < 5: window_length = 5

    true_height = ATHLETE_DB.get(athlete_key, {"height": 1.65})["height"]

    coordinate_data = []
    frame_count = 0

    # Vraćamo static_image_mode=False za tečno praćenje, ali dižemo confidence da sprečimo greške
    with mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                      min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_world_landmarks:
                for id, (lm, lm_world) in enumerate(zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark)):
                    coordinate_data.append({
                        "Frame": frame_count, 
                        "Time": frame_count * dt,
                        "FPS": fps,
                        "Landmark_ID": id, 
                        "Visibility": lm.visibility,
                        "X_raw": lm_world.x, 
                        "Y_raw": lm_world.y, 
                        "Z_raw": lm_world.z
                    })
            frame_count += 1
            print(f" -> Detekcija: {frame_count}/{total_frames} frejmova", end="\r")

    cap.release()

    if not coordinate_data:
        print(f"\n -> Nema koordinata za {athlete_key}.")
        continue

    # ==============================================================
    # FAZA 2: POST-PROCESIRANJE (Naučni standard)
    # ==============================================================
    df = pd.DataFrame(coordinate_data)

    # 1. KALIBRACIJA RAZMERE (Pretvaranje MP jedinica u prave metre)
    # Tražimo maksimalnu razdaljinu između glave (0) i stopala (31/32) da dobijemo MP visinu
    head_y = df[df['Landmark_ID'] == 0].groupby('Frame')['Y_raw'].mean()
    foot_y = df[df['Landmark_ID'].isin([31, 32])].groupby('Frame')['Y_raw'].max()
    
    # Koristimo 90. percentil visine da izbegnemo šum i momente kada su savijeni
    mp_heights = (foot_y - head_y).dropna()
    estimated_mp_height = np.percentile(mp_heights, 90) if not mp_heights.empty else 1.0
    scale_factor = true_height / estimated_mp_height

    # Primenjujemo kalibraciju na sirove podatke
    for col in ['X_raw', 'Y_raw', 'Z_raw']:
        df[col] = df[col] * scale_factor

    # Inicijalizujemo 'Clean' kolone
    df['X_clean'] = df['X_raw']
    df['Y_clean'] = df['Y_raw']
    df['Z_clean'] = df['Z_raw']
    df['Detection_OK'] = True

    # 2. OUTLIER REMOVAL (Uklanjanje fizički nemogućih skokova i slabe vidljivosti)
    # Ako je visibility manji od 0.3, smatramo da kamera nije sigurna
    bad_vis = df['Visibility'] < 0.3
    df.loc[bad_vis, ['X_clean', 'Y_clean', 'Z_clean']] = np.nan
    df.loc[bad_vis, 'Detection_OK'] = False

    # Provera brzine: ako zglob "skoči" brzinom većom od 15 m/s u jednom frejmu (fizički nemoguće), brišemo ga
    for lm_id in range(33):
        idx = df['Landmark_ID'] == lm_id
        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            # Računamo brzinu promene koordinata (m/s)
            diff = df.loc[idx, col].diff().abs() / dt
            # Prag od 15 m/s (najbrži udarci u sportu su oko 10-15 m/s)
            outliers = diff > 15.0
            df.loc[idx & outliers, col] = np.nan
            df.loc[idx & outliers, 'Detection_OK'] = False

    # 3. INTERPOLACIJA ZASNOVANA NA LIMITU (MAX 0.5 sekundi, tj. oko 15 frejmova)
    # Nećemo da izmišljamo podatke ako klizačica nestane iz kadra na duže vreme!
    max_gap_frames = int(fps * 0.5) 
    df[['X_clean', 'Y_clean', 'Z_clean']] = df.groupby('Landmark_ID')[['X_clean', 'Y_clean', 'Z_clean']].transform(
        lambda x: x.interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
    )

    # 4. ADAPTIVNI SAVITZKY-GOLAY FILTER (Samo na očišćene podatke)
    def apply_savgol(series):
        # Ako i dalje ima NaN vrednosti (rupe veće od 0.5s), SavGol puca. Zato popunjavamo privremeno.
        s_filled = series.ffill().bfill() 
        if len(s_filled) > window_length:
            smoothed = savgol_filter(s_filled, window_length=window_length, polyorder=2)
            # Vraćamo NaN tamo gde je originalno bila velika rupa (da ne lažiramo fiziku)
            smoothed[series.isna()] = np.nan
            return smoothed
        return series

    for col in ['X_clean', 'Y_clean', 'Z_clean']:
        df[col] = df.groupby('Landmark_ID')[col].transform(apply_savgol)

    # Spremanje u CSV
    csv_filename = f"koordinate_{athlete_key}_naucan.csv"
    csv_save_path = os.path.join(output_folder, csv_filename)
    # Sortiranje kolona radi bolje preglednosti
    cols = ['Frame', 'Time', 'FPS', 'Landmark_ID', 'Visibility', 'Detection_OK',
            'X_raw', 'Y_raw', 'Z_raw', 'X_clean', 'Y_clean', 'Z_clean']
    df = df[cols]
    df.to_csv(csv_save_path, index=False)
    
    print(f"\n -> USPEH: Sačuvano u '{csv_filename}'")
    print(f" -> Skalirano sa visinom {true_height}m (Faktor: {scale_factor:.2f})")
    print(f" -> SavGol prozor: {window_length} frejmova.")

print("\n" + "="*85)
print(f" SVI PODACI (RAW + CLEAN) SU SAČUVANI U:\n {output_folder}")
print(" ZAVRŠENO!")
print("="*85)
