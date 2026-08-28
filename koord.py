import os
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

#ovde ce mi ici moji izlazne koordinatew kada ih izvucem iz videa
base_dir = r"C:\Users\PC\Videos\Screen Recordings"

# putanja do modela
model_path = os.path.join(base_dir, "stabilnost_piruete_cnn.h5")
scaler_path = os.path.join(base_dir, "scaler.pkl")

# Provera postojanja modela i skalera na disku
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Greška: U navedenom folderu nije pronađen model 'stabilnost_piruete_cnn.h5' ili skaler 'scaler.pkl'!")

# video koji se analizira
video_naziv = "marianelanunezbalet.mp4" 

video_input_path = os.path.join(base_dir, video_naziv)
video_output_path = os.path.join(base_dir, "analiza_" + video_naziv)

if not os.path.exists(video_input_path):
    raise FileNotFoundError(f"Greška: U folderu nije pronađen video fajl pod nazivom {video_naziv}!")

# Učitavanje neuronske mreže i skalera
print("Učitavanje neuronske mreže i kineziološkog skalera...")
model = tf.keras.models.load_model(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Konstante za proračun fizike iz tvog rada
h = 1.74 # Marianela Nuñez je visoka 1.74m
m = 52.0 # Marianela Nuñez ima 52kg
a_type = "balet" # "balet" ili "klizanje"
baza_r = 0.05 if a_type == "balet" else 0.08
g_const = 9.81
FPS = 30
dt = 1.0 / FPS

# Inicijalizacija MediaPipe-a
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(video_input_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, FPS, (width, height))

# Bafer za krive kretanja
all_thetas = []
all_x_coms = []
frame_idx = 0

print("Započeto kineziološko praćenje i rano predviđanje nestabilnosti u realnom vremenu...")

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    
    # Konverzija boja za MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Srednji položaj kukova i ramena za osu i nagib
        x_axis = (landmarks[23].x + landmarks[24].x) / 2
        z_axis = (landmarks[23].z + landmarks[24].z) / 2
        
        # Proračun ugla rotacije ramena
        dx = landmarks[12].x - landmarks[11].x
        dz = landmarks[12].z - landmarks[11].z
        theta_rot = np.arctan2(dz, dx)
        all_thetas.append(theta_rot)
        
        # Proračun unwrapped ugla i ugaonih izvoda u hodu
        theta_unwrapped = np.unwrap(all_thetas)[-1]
        if frame_idx > 0:
            omega = (theta_unwrapped - prev_theta_unwrapped) / dt
            alpha = (omega - prev_omega) / dt if 'prev_omega' in locals() else 0.0
        else:
            omega = 0.0
            alpha = 0.0
            
        prev_theta_unwrapped = theta_unwrapped
        prev_omega = omega
        
        # Proračun položaja CoM i stopala
        x_com_weighted = np.mean([lm.x for lm in landmarks]) * h
        y_com_weighted = np.mean([lm.y for lm in landmarks]) * h
        all_x_coms.append(x_com_weighted)
        
        y_foot_l = landmarks[31].y; y_foot_r = landmarks[32].y
        if y_foot_l > y_foot_r:
            y_foot = y_foot_l * h; x_foot = landmarks[31].x * h
        else:
            y_foot = y_foot_r * h; x_foot = landmarks[32].x * h
            
        l_pendulum = abs(y_com_weighted - y_foot)
        if l_pendulum < 0.1: l_pendulum = 0.8
        omega_0 = np.sqrt(g_const / l_pendulum)
        
        # Brzina kretanja CoM
        v_com = (x_com_weighted - all_x_coms[-2]) / dt if frame_idx > 1 else 0.0
        
        r_com = abs(x_foot - x_com_weighted)
        tilt = np.degrees(np.arcsin(np.clip(r_com / l_pendulum, 0.0, 1.0)))
        theta_crit = np.degrees(np.arcsin(baza_r / l_pendulum))
        
        # Ekstrapolirani CoM i MoS
        X_CoM = x_com_weighted + (v_com / omega_0)
        MoS = baza_r - abs(x_foot - X_CoM)
        
        # --- PREDVIĐANJE PADA PREKO TIME-TO-BOUNDARY (TTB) ---
        rastojanje_do_ivice = baza_r - abs(x_foot - x_com_weighted)
        if abs(v_com) > 0.001:
            TTB = rastojanje_do_ivice / abs(v_com)
        else:
            TTB = 99.0  # stabilno
            
        # Boja skeleta (Zelena za stabilno, Crvena za rani alarm pada)
        boja_skeleta = (0, 255, 0)
        tekst_alarma = "STABILNO"
        
        # AKO TTB padne ispod 0.4 sekunde, palimo rano kineziološko upozorenje
        if TTB < 0.4 and v_com != 0:
            boja_skeleta = (0, 0, 255) # Crvena u BGR formatu
            tekst_alarma = f"ALARM: Pad za {TTB:.2f} s!"
            
        # Iscrtavanje teksta i očitavanja na frejmu
        cv2.putText(image, tekst_alarma, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, boja_skeleta, 3)
        cv2.putText(image, f"Ugao nagiba: {tilt:.2f} stepeni", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"TTB (Vreme do pada): {TTB:.2f} s", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Iscrtavanje stabilnog kineziološkog skeleta
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    out.write(image)
    frame_idx += 1

cap.release()
out.release()
pose.close()
print(f"\nAnaliza je uspešno završena! Rezultujući video je sačuvan na tvoj disk: {video_output_path}")