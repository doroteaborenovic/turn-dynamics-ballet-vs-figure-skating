import os
import glob
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def calculate_angle_3d(a, b, c):
    """Računa ugao u zglobovima B u 3D prostoru (teme B)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return np.nan
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video_folder(folder_path, output_dir="kinematika_rezultati"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pronalazi sve MP4 i AVI fajlove u folderu
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not video_files:
        print(f"Nije pronađen nijedan video u: {folder_path}")
        return

    print(f"Pronađeno videa za obradu: {len(video_files)}")

    mp_pose = mp.solutions.pose

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"\nObrada videa: {video_name}...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0  # podrazumevani FPS ako video nema podatak
        dt = 1.0 / fps

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        data_rows = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            row = [frame_idx, frame_idx * dt]

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Snimanje svih 33 tačaka
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                # Primer: Izračunavanje ugla desnog lakta (Rame=12, Lakat=14, Šaka=16)
                p12 = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
                p14 = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
                p16 = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
                
                elbow_angle = calculate_angle_3d(p12, p14, p16)
                row.append(elbow_angle)
            else:
                # Ako čovek nije detektovan
                row.extend([np.nan] * (33 * 4 + 1))

            data_rows.append(row)
            frame_idx += 1

        cap.release()
        pose.close()

        # Definisane kolona
        columns = ["frame", "timestamp_sec"]
        for i in range(33):
            columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"])
        columns.append("angle_right_elbow")

        df = pd.DataFrame(data_rows, columns=columns)

        # Izračunavanje ugaone brzine i ubrzanja (uz filtriranje)
        angles = df["angle_right_elbow"].interpolate().bfill().ffill()
        
        if len(angles) >= 7:
            # Filtriranje šuma (Savitzky-Golay filter)
            smooth_angles = savgol_filter(angles, window_length=7, polyorder=2)
            df["angle_right_elbow_smooth"] = smooth_angles
            
            # Ugaona brzina (deg/s)
            df["angular_velocity_deg_s"] = np.gradient(smooth_angles, dt)
            # Ugaono ubrzanje (deg/s^2)
            df["angular_acceleration_deg_s2"] = np.gradient(df["angular_velocity_deg_s"], dt)

        # Čuvanje u CSV
        csv_name = os.path.splitext(video_name)[0] + "_kinematics.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"Završeno! Rezultat sačuvan u: {csv_path}")

folder_videa = r"C:\Users\PC\Videos\Screen Recordings\videiples"
process_video_folder(folder_videa)
#metod ovde je bio da se izvuce 33 kljucne tacke tela pomocu emdia pipe pose modela i koristi se oristiš najpreciznijua verzija mediapipe modelaa
# #media pipe je google-ova biblioteka za obradu slike i videa koja omogućava detekciju i praćenje ljudskog tela, ruku, lica i drugih objekata u realnom vremenu. U ovom kodu, koristi se za detekciju 33 ključne tačke ljudskog tela (landmarke) u videu, što omogućava analizu kinematike pokreta sportista.
#dakle ja ucitam video, krositim googleovu biblioteku za ekstrakciju koordianta koja gleda frejm po frejm i detektuje gde je osoba pa tako i psotavlja 33 kljcune tacke i cuvam x y z koordianta od cega se z "nagadja" koja je dubina slike
#  onda kada negde dodje do greske/pogesno ucitane koordiante onda se koristi interpollacina da bi se izejdnacio deo gde je doslo do nekog prekida ili seckanja u videu  i koristi se golay filter da bis euklonio sum i kao da se koordiante tresu i da bi se cikica stabilizovao (ugoana brzina i ubrzanje se kasnije rade kod mene ponovo)