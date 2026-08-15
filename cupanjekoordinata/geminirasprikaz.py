import os
import cv2
import pandas as pd
import numpy as np
import time

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12)
]

input_dir = "kinematika_rezultati"
output_video_dir = "konacni_videi"

os.makedirs(output_video_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

for csv_file in csv_files:
    athlete_key = csv_file.replace("_kinematics.csv", "").replace("_konacne.csv", "")
    csv_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(csv_path)

    # Određivanje tačnog FPS-a na osnovu timestamp kolone (ako postoji) za 100% realnu brzinu
    if 'timestamp_sec' in df.columns and len(df) > 1:
        time_diff = df['timestamp_sec'].iloc[1] - df['timestamp_sec'].iloc[0]
        fps = round(1.0 / time_diff) if time_diff > 0 else 30.0
    elif 'FPS' in df.columns:
        fps = float(df['FPS'].iloc[0])
    else:
        fps = 30.0  # Podrazumevani FPS

    frame_delay = 1.0 / fps  # Tačno vreme koliko jedan frejm treba da traje u sekundama

    width, height = 1280, 720
    video_path = os.path.join(output_video_dir, f"video_{athlete_key}_konacno.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Auto-skaliranje
    all_x, all_y = [], []
    for i in range(33):
        if f"x_{i}" in df.columns and f"y_{i}" in df.columns:
            all_x.extend(df[f"x_{i}"].dropna())
            all_y.extend(df[f"y_{i}"].dropna())

    if len(all_x) > 0 and len(all_y) > 0:
        x_min, x_max = np.percentile(all_x, 2), np.percentile(all_x, 98)
        y_min, y_max = np.percentile(all_y, 2), np.percentile(all_y, 98)
        
        span_x = max(x_max - x_min, 0.2)
        span_y = max(y_max - y_min, 0.2)

        scale_x = (width * 0.65) / span_x
        scale_y = (height * 0.65) / span_y
        scale = min(scale_x, scale_y)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
    else:
        scale = 500
        center_x, center_y = 0.5, 0.5

    print(f"\nPrikazujem i snimam: {athlete_key} na {fps} FPS...")

    has_landmark_id = 'Landmark_ID' in df.columns
    total_frames = int(df['frame'].max()) + 1 if 'frame' in df.columns else len(df)

    for frame_idx in range(total_frames):
        t_start = time.time()  # Merenje vremena za precizan prikaz

        canvas = np.ones((height, width, 3), dtype=np.uint8) * 18
        
        for x in range(0, width, 80):
            cv2.line(canvas, (x, 0), (x, height), (28, 28, 28), 1)
        for y in range(0, height, 80):
            cv2.line(canvas, (0, y), (width, y), (28, 28, 28), 1)

        coords = {}

        if has_landmark_id:
            frame_data = df[df['frame'] == frame_idx]
            for _, row in frame_data.iterrows():
                lm_id = int(row['Landmark_ID'])
                x_val = row.get('X_clean', row.get('x', np.nan))
                y_val = row.get('Y_clean', row.get('y', np.nan))
                if not np.isnan(x_val) and not np.isnan(y_val):
                    px = int((x_val - center_x) * scale + width / 2)
                    py = int((y_val - center_y) * scale + height / 2)
                    coords[lm_id] = (px, py)
        else:
            row = df.iloc[frame_idx]
            for i in range(33):
                x_val, y_val = row.get(f"x_{i}", np.nan), row.get(f"y_{i}", np.nan)
                if not np.isnan(x_val) and not np.isnan(y_val):
                    px = int((x_val - center_x) * scale + width / 2)
                    py = int((y_val - center_y) * scale + height / 2)
                    coords[i] = (px, py)

        # 1. Crtanje kostiju
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in coords and end_idx in coords:
                pt1, pt2 = coords[start_idx], coords[end_idx]
                cv2.line(canvas, pt1, pt2, (255, 120, 0), 5, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, (255, 220, 100), 2, cv2.LINE_AA)

        # 2. Crtanje zglobova
        for idx, pt in coords.items():
            cv2.circle(canvas, pt, 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Informacioni panel
        timestamp = frame_idx * frame_delay
        cv2.rectangle(canvas, (20, 20), (380, 100), (35, 35, 35), -1)
        cv2.rectangle(canvas, (20, 20), (380, 100), (80, 80, 80), 1)

        cv2.putText(canvas, f"Atleticar: {athlete_key[:18]}", (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frejm: {frame_idx} | Brzina: {fps:.0f} FPS", (35, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(canvas)
        cv2.imshow("Prikaz u realnoj brzini (Pritisni 'q' za izlaz)", canvas)

        # Održavanje realnog vremena prikaza na ekranu
        elapsed = time.time() - t_start
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    out.release()

cv2.destroyAllWindows()
print("\nSvi videi su izgenerisani u realnoj brzini!")
