import pandas as pd
import numpy as np
import cv2
import os
from scipy.interpolate import PchipInterpolator

def pchip_interpolate_gaps(values, max_gap=6):
    values = np.asarray(values, dtype=float)
    result = values.copy()
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return result
    
    x = np.arange(len(values))
    missing_indices = np.where(~valid)[0]
    if len(missing_indices) == 0:
        return result

    blocks, start, prev = [], missing_indices[0], missing_indices[0]
    for idx in missing_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            blocks.append((start, prev))
            start, prev = idx, idx
    blocks.append((start, prev))

    interpolator = PchipInterpolator(x[valid], values[valid], extrapolate=False)
    for start, end in blocks:
        if (end - start + 1) <= max_gap:
            indices = np.arange(start, end + 1)
            interp_vals = interpolator(indices)
            valid_interp = np.isfinite(interp_vals)
            result[indices[valid_interp]] = interp_vals[valid_interp]
            
    return result

def clean_and_interpolate_coordinates(df):
    df = df.copy()
    fps = float(df['FPS'].iloc[0]) if 'FPS' in df.columns and not df['FPS'].isna().all() else 30.0
    
    min_frame, max_frame = int(df['Frame'].min()), int(df['Frame'].max())
    all_frames = np.arange(min_frame, max_frame + 1)
    
    cleaned_groups = []
    for landmark_id in sorted(df['Landmark_ID'].unique()):
        group = df[df['Landmark_ID'] == landmark_id].copy()
        group = group.drop_duplicates(subset=['Frame'], keep='first').set_index('Frame')
        group = group.reindex(all_frames)
        group.index.name = 'Frame'
        group['Landmark_ID'] = landmark_id
        group['FPS'] = fps
        group['Time'] = group.index.to_numpy() / fps

        for col in ['X_clean', 'Y_clean', 'Z_clean']:
            vals = np.array(pd.to_numeric(group[col], errors='coerce').values, dtype=float)
            diffs = np.abs(np.diff(vals, prepend=vals[0]))
            vals[diffs > 0.2] = np.nan
            vals = pchip_interpolate_gaps(vals, max_gap=8)
            group[col] = vals

        cleaned_groups.append(group.reset_index())
        
    result = pd.concat(cleaned_groups, ignore_index=True).sort_values(['Frame', 'Landmark_ID']).reset_index(drop=True)
    return result

# Fiksne veze MediaPipe skeleta - NIKADA SE NE MENJAJU
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12)
]

input_dir = "konacne_koordinate"
output_video_dir = "konacni_videi"
output_coord_dir = "koordinate_sredjene"

os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_coord_dir, exist_ok=True)

print("\nPokrećem završnu obradu i generisanje stabilnih videa...")

for csv_file in os.listdir(input_dir):
    if not csv_file.endswith("_konacne.csv"): continue
    
    athlete_key = csv_file.replace("_konacne.csv", "")
    csv_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    if 'X_clean' not in df.columns or 'Landmark_ID' not in df.columns:
        continue

    df_stabilized = clean_and_interpolate_coordinates(df)
    
    coord_out_path = os.path.join(output_coord_dir, f"{athlete_key}_sredjene.csv")
    df_stabilized.to_csv(coord_out_path, index=False)
    
    fps = float(df_stabilized['FPS'].iloc[0])
    width, height = 1280, 720
    video_path = os.path.join(output_video_dir, f"video_{athlete_key}_konacno.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Automatsko izračunavanje optimalne skale da ceo skelet stane u kadar bez prevelikog zumiranja
    valid_x = df_stabilized['X_clean'].dropna()
    valid_y = df_stabilized['Y_clean'].dropna()
    
    if len(valid_x) > 0 and len(valid_y) > 0:
        x_min, x_max = np.percentile(valid_x, 1), np.percentile(valid_x, 99)
        y_min, y_max = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
        span_x = max(x_max - x_min, 0.5)
        span_y = max(y_max - y_min, 0.5)
        
        # Prilagođavanje skale dimenzijama prozora (1280x720) sa marginom
        scale_x = (width * 0.65) / span_x
        scale_y = (height * 0.65) / span_y
        scale = min(scale_x, scale_y)
    else:
        scale = 300.0

    for frame_idx in sorted(df_stabilized['Frame'].unique()):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frame_data = df_stabilized[df_stabilized['Frame'] == frame_idx]
        curr_time = frame_data['Time'].iloc[0]
        
        # Centriranje kadra na sredinu tela (prosek svih vidljivih tačaka u tom frejmu)
        h_x = frame_data['X_clean'].dropna().mean()
        h_y = frame_data['Y_clean'].dropna().mean()
        if np.isnan(h_x) or np.isnan(h_y):
            h_x, h_y = 0.0, 0.0

        points_2d = {}
        for _, row in frame_data.iterrows():
            lm_id = int(row['Landmark_ID'])
            x, y = row['X_clean'], row['Y_clean']
            if np.isfinite(x) and np.isfinite(y):
                # Standardna 2D projekcija sa stabilnim centriranjem i srazmerom
                px = int(width / 2 + (x - h_x) * scale)
                py = int(height / 2 - (y - h_y) * scale) # obrnuto Y za ekran
                points_2d[lm_id] = (px, py)
                cv2.circle(canvas, (px, py), 4, (255, 220, 0), -1)

        # Crtanje fiksnih veza bez ikakve mogućnosti deformacije skeleta
        for p1, p2 in POSE_CONNECTIONS:
            if p1 in points_2d and p2 in points_2d:
                cv2.line(canvas, points_2d[p1], points_2d[p2], (0, 255, 100), 2, cv2.LINE_AA)
                
        cv2.putText(canvas, f"Atleta: {athlete_key.upper()} | Time: {curr_time:.2f}s | Frame: {frame_idx}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(canvas)
        
    out.release()
    print(f"  ✅ Uspešno sačuvano -> Video: {video_path}")

print("\n=== SVE JE ZAVRŠENO I SPREMNO ===")