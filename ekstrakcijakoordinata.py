import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Tačna putanja do videa iz kojih se vade koordinate
base_path = r"C:\Users\PC\Videos\Screen Recordings"
output_folder = os.path.join(base_path, "processed_coordinates")
os.makedirs(output_folder, exist_ok=True)

# Spisak fajlova za obradu
videos_to_process = [
    {"filename": "viktoriakapitonovabalet.mp4", "key": "kapitonova"},
    {"filename": "marianelanunezbalet.mp4", "key": "marianela"},
    {"filename": "alysaliuklizanjemp4.mp4", "key": "liu"},
    {"filename": "anascerebakovaklizanje.mp4", "key": "scerebakova"},
    {"filename": "aleksandratrusovaklizanjep4.mp4", "key": "trusova"},
    {"filename": "mariakhorevabalet.mp4", "key": "khoreva"}
]

print("="*75)
print("      POKREĆEM EKSTRAKCIJU KOORDINATA (STABILNI BEZ-PROZORSKI MOD)")
print(f" Izvor folder:  {base_path}")
print(f" Ciljni folder: {output_folder}")
print("="*75)

# Provera da li izvorna putanja postoji
if not os.path.exists(base_path):
    print(f"GREŠKA: Folder '{base_path}' ne postoji! Proverite putanju.")
    sys.exit()

successful_count = 0

# Prolazak kroz svaki video sa spiska
for index, item in enumerate(videos_to_process, 1):
    video_filename = item["filename"]
    athlete_key = item["key"]
    
    full_video_path = os.path.join(base_path, video_filename)
    
    # Provera postojanja fajla
    if not os.path.exists(full_video_path):
        print(f"\n[{index}/6] PRESKAČEM: '{video_filename}' (fajl nije pronađen)")
        continue
        
    print(f"\n[{index}/6] OBRAĐUJEM: '{video_filename}' ({athlete_key})...")
    cap = cv2.VideoCapture(full_video_path)

    if not cap.isOpened():
        print(f" -> GREŠKA: Ne mogu da otvorim video fajl!")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  
    dt = 1.0 / fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" -> Detektovan FPS: {fps:.2f} | Ukupno frejmova: {total_frames}")

    coordinate_data = []
    frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,         
        enable_segmentation=False,   
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            current_time = frame_count * dt

            # Konverzija frejma u RGB za MediaPipe jer on tako prepoznaje 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Ako su uspešno detektovani zglobovi, upisujemo ih u listu
            if results.pose_landmarks and results.pose_world_landmarks:
                for id, (lm, lm_world) in enumerate(zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark)):
                    coordinate_data.append({
                        "Frame": frame_count,
                        "Time": current_time,           
                        "Landmark_ID": id,
                        "X_world": lm_world.x,          
                        "Y_world": lm_world.y,
                        "Z_world": lm_world.z,
                        "Visibility": lm.visibility,     
                        "X_pixel": lm.x,                
                        "Y_pixel": lm.y,
                        "Z_pixel": lm.z  #procena dubine u pikselima, mada nije standardno i može biti neprecizno i onda su x i z normalizovane i idu u odnosu na sliku tj video ali se gleda svaki frejm
                    })

            frame_count += 1
            
            # Ispis napretka u terminalu (osvežava se u istoj liniji)
            if frame_count % 50 == 0 or frame_count == total_frames:
                pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"    * Napredak: {frame_count}/{total_frames} frejmova ({pct:.1f}%)", end="\r")

        print()  # Prelazak u novi red nakon što se video završi

    cap.release()

    # Čuvanje sakupljenih koordinata u CSV fajl
    if len(coordinate_data) > 0:
        df_coords = pd.DataFrame(coordinate_data)
        csv_filename = f"koordinate_{athlete_key}.csv"
        csv_save_path = os.path.join(output_folder, csv_filename)
        df_coords.to_csv(csv_save_path, index=False)
        print(f" -> USPEH: Koordinate sačuvane u: '{csv_filename}'")
        successful_count += 1
    else:
        print(f" -> UPOZORENJE: Nisu detektovane koordinate za ovaj video.")

print("\n" + "="*75)
print(f" OBRADA ZAVRŠENA! Uspešno obrađeno: {successful_count}/6 video snimaka.")
print(f" Svi generisani .csv fajlovi se nalaze u folderu:\n {output_folder}")
print("="*75)
