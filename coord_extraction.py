#ekstrakcija koordinata u izlazni folder na mom lapotpu 
import os
import glob
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class OneEuroFilter:
    """
    Casiez, Roussel, Vogel (2012) - "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems"

    Prednost nad Savitzky-Golay/Butterworth: adaptivno menja jačinu
    filtriranja u zavisnosti od brzine promene signala. Kad se telo
    kreće sporo -> jako glača (uklanja jitter). Kad se kreće brzo
    (npr. tokom same rotacije) -> manje glača (ne unosi lag/kašnjenje
    koje bi lažno smanjilo izmerenu ugaonu brzinu).
    """

    def __init__(self, freq, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def make_filter_bank(n_landmarks, freq, min_cutoff=1.0, beta=0.02):
    """Po jedan OneEuroFilter za svaku (landmark, osu) kombinaciju."""
    return {
        (lm, ax): OneEuroFilter(freq, min_cutoff=min_cutoff, beta=beta)
        for lm in range(n_landmarks) for ax in ('x', 'y', 'z')
    }

def bbox_from_landmarks(landmarks_px, frame_w, frame_h, margin_ratio=0.35):
    """Bounding box oko detektovanih markera + margina, u piksel koordinatama."""
    xs = [p[0] for p in landmarks_px if p is not None]
    ys = [p[1] for p in landmarks_px if p is not None]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    mx = w * margin_ratio
    my = h * margin_ratio
    x_min = max(0, int(x_min - mx))
    y_min = max(0, int(y_min - my))
    x_max = min(frame_w, int(x_max + mx))
    y_max = min(frame_h, int(y_max + my))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def crop_to_square_min_size(bbox, frame_w, frame_h, min_size=480):
    """Proširi bbox na kvadrat od bar min_size px, centriran, unutar granica frejma."""
    x_min, y_min, x_max, y_max = bbox
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    side = max(x_max - x_min, y_max - y_min, min_size)
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(frame_w, cx + half)
    y1 = min(frame_h, cy + half)
    # ako je isekao ivicu frejma, pomeri nazad da zadrži veličinu
    if x1 - x0 < side:
        if x0 == 0:
            x1 = min(frame_w, x0 + side)
        else:
            x0 = max(0, x1 - side)
    if y1 - y0 < side:
        if y0 == 0:
            y1 = min(frame_h, y0 + side)
        else:
            y0 = max(0, y1 - side)
    return (x0, y0, x1, y1)


#ovde ide glavna obrada
def process_video_folder(folder_path, output_dir="kinematika_rezultati_v2",
                          use_roi_tracking=True, apply_one_euro=True,
                          one_euro_min_cutoff=1.2, one_euro_beta=0.03):

    os.makedirs(output_dir, exist_ok=True)

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
            fps = 30.0
        dt = 1.0 / fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Dva modela: jedan za full-frame (kad se izgubi trag), jedan za ROI crop
        pose_full = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose_roi = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.4,   # niži prag OK jer je ROI zumiran = čistiji signal
            min_tracking_confidence=0.4
        )

        n_lm = 33
        filter_bank = make_filter_bank(n_lm, fps, one_euro_min_cutoff, one_euro_beta) if apply_one_euro else None

        data_rows = []
        world_rows = []
        quality_rows = []

        current_bbox = None  # (x0,y0,x1,y1) u px prethodnog frejma, None = koristi full frame
        frame_idx = 0
        lost_track_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            use_roi = use_roi_tracking and current_bbox is not None
            if use_roi:
                x0, y0, x1, y1 = current_bbox
                crop = frame[y0:y1, x0:x1]
                image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = pose_roi.process(image_rgb)
                crop_w, crop_h = (x1 - x0), (y1 - y0)
                offset_x, offset_y = x0, y0
            else:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_full.process(image_rgb)
                crop_w, crop_h = frame_w, frame_h
                offset_x, offset_y = 0, 0

            row = [frame_idx, frame_idx * dt]
            world_row = [frame_idx, frame_idx * dt]

            detected = results.pose_landmarks is not None and results.pose_world_landmarks is not None

            if detected:
                landmarks_norm = results.pose_landmarks.landmark
                landmarks_world = results.pose_world_landmarks.landmark

                # px koordinate u punom frejmu (za ROI bbox sledećeg frejma)
                landmarks_px = []
                vis_list = []

                for i in range(n_lm):
                    lm = landmarks_norm[i]
                    # mapiranje nazad iz crop-a u koordinate punog frejma
                    px = offset_x + lm.x * crop_w
                    py = offset_y + lm.y * crop_h
                    landmarks_px.append((px, py))
                    vis_list.append(lm.visibility)

                    # normalizovano na PUN frejm (konzistentno bez obzira na ROI/full)
                    x_full = px / frame_w
                    y_full = py / frame_h

                    if apply_one_euro:
                        t = frame_idx * dt
                        x_full = filter_bank[(i, 'x')].filter(x_full, t)
                        y_full = filter_bank[(i, 'y')].filter(y_full, t)
                        z_f = filter_bank[(i, 'z')].filter(lm.z, t)
                    else:
                        z_f = lm.z

                    row.extend([x_full, y_full, z_f, lm.visibility])

                    wlm = landmarks_world[i]
                    world_row.extend([wlm.x, wlm.y, wlm.z, wlm.visibility])

                mean_vis = float(np.mean(vis_list))
                n_low_vis = int(np.sum(np.array(vis_list) < 0.5))

                # ažuriraj ROI za sledeći frejm
                if use_roi_tracking:
                    bbox = bbox_from_landmarks(landmarks_px, frame_w, frame_h, margin_ratio=0.45)
                    if bbox is not None:
                        current_bbox = crop_to_square_min_size(bbox, frame_w, frame_h, min_size=480)
                    lost_track_counter = 0
            else:
                row.extend([np.nan] * (n_lm * 4))
                world_row.extend([np.nan] * (n_lm * 4))
                mean_vis = 0.0
                n_low_vis = n_lm
                lost_track_counter += 1
                # posle 5 uzastopnih promašaja, vrati se na full-frame detekciju
                if lost_track_counter >= 5:
                    current_bbox = None

            quality_rows.append({
                "frame": frame_idx,
                "timestamp_sec": frame_idx * dt,
                "detected": detected,
                "used_roi": use_roi,
                "mean_visibility": mean_vis,
                "n_low_visibility_landmarks": n_low_vis
            })

            data_rows.append(row)
            world_rows.append(world_row)
            frame_idx += 1

        cap.release()
        pose_full.close()
        pose_roi.close()

        columns = ["frame", "timestamp_sec"]
        for i in range(n_lm):
            columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"])

        world_columns = ["frame", "timestamp_sec"]
        for i in range(n_lm):
            world_columns.extend([f"wx_{i}", f"wy_{i}", f"wz_{i}", f"wvis_{i}"])

        df = pd.DataFrame(data_rows, columns=columns)
        df_world = pd.DataFrame(world_rows, columns=world_columns)
        df_quality = pd.DataFrame(quality_rows)

        base_name = os.path.splitext(video_name)[0]

        csv_path = os.path.join(output_dir, f"{base_name}_kinematics.csv")
        world_csv_path = os.path.join(output_dir, f"{base_name}_world.csv")
        quality_csv_path = os.path.join(output_dir, f"{base_name}_quality.csv")

        df.to_csv(csv_path, index=False)
        df_world.to_csv(world_csv_path, index=False)
        df_quality.to_csv(quality_csv_path, index=False)

        pct_detected = df_quality["detected"].mean() * 100
        pct_low_vis = (df_quality["n_low_visibility_landmarks"] > 10).mean() * 100
        print(f"  Detektovano frejmova: {pct_detected:.1f}%")
        print(f"  Frejmova sa >10 markera niske vidljivosti: {pct_low_vis:.1f}%")
        print(f"  Sačuvano: {csv_path}")
        print(f"  Sačuvano (world/metric): {world_csv_path}")
        print(f"  Sačuvano (kvalitet): {quality_csv_path}")


if __name__ == "__main__":
    folder_videa = r"C:\Users\PC\Videos\Screen Recordings\videiples"
    process_video_folder(folder_videa)
