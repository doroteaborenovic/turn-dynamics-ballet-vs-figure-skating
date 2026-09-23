"""
RENDER "ČIKICA" (stick figure) VIDEA IZ CSV-ova ekstrakcije (MediaPipe Pose) - v2
=================================================================================
Čita folder koji pravi skripta za ekstrakciju (default: kinematika_rezultati_v2):
    <ime>_kinematics.csv  -> normalizovane x,y (0-1 na punom frejmu) + z + vis
    <ime>_world.csv       -> metrički 3D (m), centriran na kuk (opciono, ali pomaže)

i za svaki video pravi <ime>_cikice.mp4.

Šta je novo u v2 (problemi koji su se videli na prvom renderu):
  - Skelet je bio sitan jer je jedna skala važila za CEO klip (klizačica prelazi
    ceo kadar) + par pogrešnih frejmova je razvlačilo skalu.
    -> "follow" kamera: kadar prati kuk, skala je vezana za veličinu tela, pa je
       skelet uvek krupan. Mala mapa dole desno pokazuje gde je u kadru.
  - Pogrešni frejmovi (razletele kosti, markeri koji "teleportuju").
    -> pre crtanja se odbacuju: markeri van kadra, niske vidljivosti, nemoguće
       dužine kostiju (poredi se sa 3D world koordinatama) i nagli skokovi
       (Hampel filter u odnosu na pokretni medijan). Kratke rupe se popune.

Dva moda:
  --mode image  (default) - pozicija u kadru
  --mode world            - metrički, centrirano na kuk (čista kinematika tela)

Primeri:
  python render_cikice.py
  python render_cikice.py --mode world --speed 0.25
  python render_cikice.py --input "D:\\rezultati" --output "D:\\videi" --min-vis 0.25
"""

import os
import glob
import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# PODEŠAVANJA (mogu i preko argumenata komandne linije)
# ---------------------------------------------------------------------------
INPUT_DIR = "kinematika_rezultati_v2"
OUTPUT_DIR = "kinematika_videi"
VIDEOS_DIR = r"C:\Users\PC\Videos\Screen Recordings\videiples"  # originalni videi (samo za dimenzije kadra)

N_LM = 33
PAD = 0.10  # margina oko skeleta na platnu

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12),
]

BODY_IDS = [0] + list(range(11, 33))   # nos + telo (lice 1-10 se ne koristi)
TORSO = [11, 12, 23, 24]
# kosti čija se dužina proverava (glavni segmenti tela)
BONES_CHECK = [(11, 12), (23, 24), (11, 23), (12, 24), (11, 13), (13, 15),
               (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]

# BGR boje: leva strana / desna strana / središnje kosti
COL_LEFT = (255, 170, 0)
COL_RIGHT = (0, 140, 255)
COL_CENTER = (0, 255, 100)


def side_color(a, b):
    """MediaPipe: neparni indeksi = leva strana, parni = desna (za markere 11+)."""
    if a == 0 or b == 0:
        return COL_CENTER
    if a % 2 == 1 and b % 2 == 1:
        return COL_LEFT
    if a % 2 == 0 and b % 2 == 0:
        return COL_RIGHT
    return COL_CENTER


# ---------------------------------------------------------------------------
# UČITAVANJE
# ---------------------------------------------------------------------------
def _stack(df, prefix, name):
    return np.stack([df[f"{prefix}{name}_{i}"].to_numpy(dtype=float) for i in range(N_LM)], axis=1)


def load_clip(kin_path, world_path):
    kin = pd.read_csv(kin_path)
    clip = {
        "frames": kin["frame"].to_numpy(),
        "t": kin["timestamp_sec"].to_numpy(dtype=float),
        "X": _stack(kin, "", "x"),
        "Y": _stack(kin, "", "y"),
        "V": _stack(kin, "", "vis"),
        "W": None,
        "WV": None,
    }
    if world_path and os.path.exists(world_path):
        w = pd.read_csv(world_path)
        if len(w) == len(kin):
            clip["W"] = np.stack([_stack(w, "w", "x"), _stack(w, "w", "y"), _stack(w, "w", "z")], axis=2)
            clip["WV"] = _stack(w, "w", "vis")
    return clip


def find_source_dims(video_name, videos_dir):
    """Dimenzije originalnog videa (da bi x i y imali pravu proporciju)."""
    if not videos_dir or not os.path.isdir(videos_dir):
        return None
    for ext in ("mp4", "avi", "mov", "mkv"):
        path = os.path.join(videos_dir, f"{video_name}.{ext}")
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
    return None


# ---------------------------------------------------------------------------
# ČIŠĆENJE
# ---------------------------------------------------------------------------
def torso_length(X, Y, valid):
    """Medijana rastojanja sredina ramena - sredina kukova (mera veličine tela)."""
    ok = valid[:, 11] & valid[:, 12] & valid[:, 23] & valid[:, 24]
    if ok.sum() < 5:
        return None
    sh = np.stack([(X[ok, 11] + X[ok, 12]) / 2, (Y[ok, 11] + Y[ok, 12]) / 2], axis=1)
    hp = np.stack([(X[ok, 23] + X[ok, 24]) / 2, (Y[ok, 23] + Y[ok, 24]) / 2], axis=1)
    return float(np.median(np.linalg.norm(sh - hp, axis=1)))


def jump_deviation(X, Y, valid, win=5):
    """Koliko je svaki marker daleko od medijane svojih 5 susednih frejmova."""
    Xm = pd.DataFrame(np.where(valid, X, np.nan))
    Ym = pd.DataFrame(np.where(valid, Y, np.nan))
    mx = Xm.rolling(win, center=True, min_periods=3).median().to_numpy()
    my = Ym.rolling(win, center=True, min_periods=3).median().to_numpy()
    return np.hypot(Xm.to_numpy() - mx, Ym.to_numpy() - my)


def reject_bone_outliers(valid, V, P, has_world, tol, dev_n, max_bad_bones=3):
    """
    Odbaci markere čije kosti imaju nemoguću dužinu.
    world (3D metri): dužine kostiju su skoro konstantne -> dvostrana provera.
    bez world-a (2D): kosti se skraćuju usled rotacije, pa se proverava samo "predugo".
    Ako je >= max_bad_bones kostiju loše u istom frejmu, ceo frejm je pogrešan.
    """
    valid = valid.copy()
    Pm = np.where(valid[..., None], P, np.nan)
    L = np.stack([np.linalg.norm(Pm[:, a] - Pm[:, b], axis=1) for a, b in BONES_CHECK], axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if has_world:
            ratio = L / np.nanmedian(L, axis=0)
            bad = (ratio > tol) | (ratio < 1.0 / (1.25 * tol))
        else:
            bad = (L / np.nanpercentile(L, 90, axis=0)) > tol

    kill = bad.sum(axis=1) >= max_bad_bones
    valid[kill, :] = False

    # kod pojedinačno loše kosti odbaci "krivi" kraj: onaj koji učestvuje u više loših kostiju
    # (npr. lakat koji je odleteo kvari i nadlakticu i podlakticu); pri izjednačenju onaj
    # koji je skočio / ima manju vidljivost
    cnt = np.zeros(valid.shape, dtype=int)
    for b, (a, c) in enumerate(BONES_CHECK):
        cnt[:, a] += bad[:, b]
        cnt[:, c] += bad[:, b]
    score = np.nan_to_num(dev_n, nan=0.0) + (1.0 - np.nan_to_num(V, nan=0.0))
    for b, (a, c) in enumerate(BONES_CHECK):
        idx = np.flatnonzero(bad[:, b] & ~kill)
        if idx.size:
            a_worse = (cnt[idx, a] > cnt[idx, c]) | ((cnt[idx, a] == cnt[idx, c]) & (score[idx, a] >= score[idx, c]))
            valid[idx, np.where(a_worse, a, c)] = False
    return valid


def lowpass_nan_safe(v, cutoff, fs, order=4):
    """Butterworth (filtfilt) po neprekidnim segmentima bez NaN - ne razvlači rupe."""
    out = v.copy()
    b, a = butter(order, min(cutoff / (0.5 * fs), 0.99), btype="low")
    padlen = 3 * max(len(a), len(b))
    ok = (~np.isnan(v)).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok, [0]))))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s > padlen:
            out[s:e] = filtfilt(b, a, v[s:e])
    return out


def clean_clip(X, Y, V, pre_valid, W3, fps, args):
    """Vraća očišćene X, Y (NaN gde marker nije pouzdan) i izveštaj."""
    rep = {}
    present = np.isfinite(X) & np.isfinite(Y)
    rep["ukupno"] = int(present.sum())
    valid = present.copy()

    low = valid & ~(V >= args.min_vis)
    valid &= ~low
    rep["niska vidljivost"] = int(low.sum())

    oof = valid & ~pre_valid
    valid &= ~oof
    rep["van kadra"] = int(oof.sum())

    if valid.sum() == 0:
        return np.full_like(X, np.nan), np.full_like(Y, np.nan), rep

    scale = torso_length(X, Y, valid)
    if scale is None or scale <= 0:
        fx, fy = X[valid], Y[valid]
        scale = max(((np.percentile(fx, 95) - np.percentile(fx, 5)) +
                     (np.percentile(fy, 95) - np.percentile(fy, 5))) / 6.0, 1e-9)
    dev_n = jump_deviation(X, Y, valid) / scale

    if args.bone_tol > 0:
        before = valid.sum()
        P = W3 if W3 is not None else np.stack([X, Y], axis=2)
        valid = reject_bone_outliers(valid, V, P, W3 is not None, args.bone_tol, dev_n)
        rep["nemoguće kosti"] = int(before - valid.sum())

    if args.jump > 0:
        j = valid & (dev_n > args.jump)
        valid &= ~j
        rep["skokovi"] = int(j.sum())

    Xc = np.where(valid, X, np.nan)
    Yc = np.where(valid, Y, np.nan)

    # popuni samo KRATKE rupe (do max_gap frejmova)
    if args.max_gap > 0:
        Xc = np.array(pd.DataFrame(Xc).interpolate(limit=args.max_gap, limit_area="inside"), dtype=float)
        Yc = np.array(pd.DataFrame(Yc).interpolate(limit=args.max_gap, limit_area="inside"), dtype=float)

    # opciono dodatno glađenje (default isključeno - podaci su već One Euro filtrirani)
    if args.butter > 0:
        for i in range(N_LM):
            Xc[:, i] = lowpass_nan_safe(Xc[:, i], args.butter, fps)
            Yc[:, i] = lowpass_nan_safe(Yc[:, i], args.butter, fps)

    # frejm sa premalo markera nije skelet - ne crtaj krhotine
    weak = np.isfinite(Xc[:, BODY_IDS]).sum(axis=1) < args.min_points
    Xc[weak, :] = np.nan
    Yc[weak, :] = np.nan
    rep["frejmova bez skeleta"] = int(weak.sum())
    return Xc, Yc, rep


# ---------------------------------------------------------------------------
# KAMERA / SKALA
# ---------------------------------------------------------------------------
def _row_mean(A):
    ok = np.isfinite(A)
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, A, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)


def follow_center(X, Y, fps, smooth_sec):
    """Centar kadra po frejmu = sredina trupa (glatko), rupe se interpolišu."""
    out = []
    for A in (X, Y):
        c = _row_mean(A[:, TORSO])
        c = np.where(np.isfinite(c), c, _row_mean(A[:, BODY_IDS]))
        c = pd.Series(c).interpolate(limit_direction="both").to_numpy()
        if smooth_sec > 0:
            c = gaussian_filter1d(c, sigma=max(smooth_sec * fps, 0.5), mode="nearest")
        out.append(c)
    return out[0], out[1]


def build_view(X, Y, fps, view, args):
    W, H = args.width, args.height
    Xb, Yb = X[:, BODY_IDS], Y[:, BODY_IDS]
    m = np.isfinite(Xb) & np.isfinite(Yb)
    T = X.shape[0]

    if view == "follow":
        cx, cy = follow_center(X, Y, fps, args.cam_smooth)
        ext_x = max(np.percentile(np.abs(Xb - cx[:, None])[m], 98), 1e-9)
        ext_y = max(np.percentile(np.abs(Yb - cy[:, None])[m], 98), 1e-9)
        scale = min(W * (0.5 - PAD) / ext_x, H * (0.5 - PAD) / ext_y)
    else:
        x_lo, x_hi = np.percentile(Xb[m], [0.5, 99.5])
        y_lo, y_hi = np.percentile(Yb[m], [0.5, 99.5])
        cx = np.full(T, (x_lo + x_hi) / 2.0)
        cy = np.full(T, (y_lo + y_hi) / 2.0)
        scale = min(W * (1 - 2 * PAD) / max(x_hi - x_lo, 1e-9),
                    H * (1 - 2 * PAD) / max(y_hi - y_lo, 1e-9))
    return cx, cy, scale


def make_minimap(cx, cy, src_w, src_h):
    """Mala mapa: putanja kuka u kadru originalnog videa."""
    r = src_w / src_h
    if r >= 1:
        mw, mh = 240, max(int(240 / r), 40)
    else:
        mh, mw = 180, max(int(180 * r), 40)
    base = np.full((mh, mw, 3), 30, dtype=np.uint8)
    tx = np.clip(cx / src_w, 0, 1) * (mw - 1)
    ty = np.clip(cy / src_h, 0, 1) * (mh - 1)
    poly = np.round(np.stack([tx, ty], axis=1)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(base, [poly], False, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.rectangle(base, (0, 0), (mw - 1, mh - 1), (90, 90, 90), 1)
    return base, tx, ty


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------
def render_one(kin_path, base, args):
    world_path = os.path.join(os.path.dirname(kin_path), f"{base}_world.csv")
    clip = load_clip(kin_path, world_path)
    t, frames, W3 = clip["t"], clip["frames"], clip["W"]

    if len(t) < 2:
        print("  Premalo frejmova, preskačem.")
        return
    fps = 1.0 / float(np.median(np.diff(t)))

    if W3 is None:
        print("  (world CSV nije nađen - provera kostiju radi samo u 2D)")

    src_w = src_h = None
    if args.mode == "world":
        if W3 is None:
            print("  Mod 'world' traži _world.csv, preskačem.")
            return
        X, Y, V = W3[..., 0].copy(), W3[..., 1].copy(), clip["WV"]
        pre_valid = np.ones(X.shape, dtype=bool)
    else:
        Xn, Yn, V = clip["X"], clip["Y"], clip["V"]
        pre_valid = (Xn >= -0.05) & (Xn <= 1.05) & (Yn >= -0.05) & (Yn <= 1.05)
        dims = find_source_dims(base, args.videos)
        if dims:
            src_w, src_h = dims
            print(f"  Dimenzije originala: {src_w}x{src_h}")
        else:
            src_w, src_h = args.aspect, 1.0
            print(f"  Original nije nađen, koristim odnos stranica {args.aspect:.3f}")
        X, Y = Xn * src_w, Yn * src_h  # x i y u istim jedinicama (inače je x razvučen)

    X, Y, rep = clean_clip(X, Y, V, pre_valid, W3, fps, args)

    total = max(rep["ukupno"], 1)
    print(f"  Markera ukupno: {rep['ukupno']} | FPS izvora: {fps:.1f}")
    for k, v in rep.items():
        if k != "ukupno":
            print(f"    odbačeno - {k}: {v}" + (f" ({100.0 * v / total:.1f}%)" if "frejmova" not in k else ""))

    if not np.isfinite(X[:, BODY_IDS]).any():
        print("  Nema nijednog pouzdanog markera, preskačem.")
        return

    view = args.view if args.view != "auto" else ("follow" if args.mode == "image" else "fixed")
    CX, CY, scale = build_view(X, Y, fps, view, args)
    PX = args.width / 2.0 + (X - CX[:, None]) * scale   # y u MediaPipe-u ide NADOLE, pa se ne okreće
    PY = args.height / 2.0 + (Y - CY[:, None]) * scale

    mini = None
    if args.mode == "image" and view == "follow":
        mini = make_minimap(CX, CY, src_w, src_h)
        mh, mw = mini[0].shape[:2]
        mx0, my0 = args.width - mw - 20, args.height - mh - 20

    out_path = os.path.join(args.output, f"{base}_cikice.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps * args.speed, (args.width, args.height))
    n_drawn = 0

    for k in range(len(t)):
        canvas = np.full((args.height, args.width, 3), 18, dtype=np.uint8)

        pts = {}
        for i in range(N_LM):
            if np.isfinite(PX[k, i]) and np.isfinite(PY[k, i]):
                pts[i] = (int(round(PX[k, i])), int(round(PY[k, i])))

        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(canvas, pts[a], pts[b], side_color(a, b), 3, cv2.LINE_AA)

        for i, pt in pts.items():
            if 0 < i < 11:
                continue
            col = COL_CENTER if i == 0 else (COL_LEFT if i % 2 == 1 else COL_RIGHT)
            cv2.circle(canvas, pt, 8 if i == 0 else 5, col, -1, cv2.LINE_AA)

        if pts:
            n_drawn += 1
            status, txt_col = f"vis {np.nanmean(V[k]):.2f} | tacaka {len(pts)}", (255, 255, 255)
        else:
            status, txt_col = "NEMA DETEKCIJE", (60, 60, 255)

        cv2.putText(canvas, f"{base} | t={t[k]:.2f}s | frame {int(frames[k])} | {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2, cv2.LINE_AA)

        if mini is not None:
            canvas[my0:my0 + mh, mx0:mx0 + mw] = mini[0]
            cv2.circle(canvas, (mx0 + int(mini[1][k]), my0 + int(mini[2][k])), 4, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, "pozicija u kadru", (mx0, my0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"  Nacrtano frejmova: {n_drawn}/{len(t)} (pogled: {view})")
    print(f"  Sačuvano: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Čikice video iz MediaPipe CSV-ova (v2)")
    ap.add_argument("--input", default=INPUT_DIR, help="folder sa *_kinematics.csv (+ *_world.csv)")
    ap.add_argument("--output", default=OUTPUT_DIR, help="folder za izlazne videe")
    ap.add_argument("--videos", default=VIDEOS_DIR, help="folder sa originalnim videima (za dimenzije kadra)")
    ap.add_argument("--mode", choices=["image", "world"], default="image")
    ap.add_argument("--view", choices=["auto", "follow", "fixed"], default="auto",
                    help="follow = kamera prati kuk (krupan skelet); fixed = jedna skala za ceo klip. "
                         "auto: image->follow, world->fixed")
    ap.add_argument("--cam-smooth", type=float, default=0.25, help="glađenje kamere u sekundama (0 = zaključana na kuk)")
    ap.add_argument("--min-vis", type=float, default=0.35, help="marker ispod ove vidljivosti se odbacuje")
    ap.add_argument("--bone-tol", type=float, default=1.5,
                    help="tolerancija dužine kosti (x medijane); 0 = isključeno")
    ap.add_argument("--jump", type=float, default=1.0,
                    help="marker dalji od ovoliko dužina trupa od medijane suseda = skok; 0 = isključeno")
    ap.add_argument("--max-gap", type=int, default=8, help="najduža rupa (u frejmovima) koja se popunjava")
    ap.add_argument("--min-points", type=int, default=8, help="minimum markera tela da bi frejm imao skelet")
    ap.add_argument("--butter", type=float, default=0.0,
                    help="Butterworth low-pass cutoff u Hz (0 = isključeno; 4 Hz zaglađuje brze pokrete)")
    ap.add_argument("--speed", type=float, default=1.0, help="brzina reprodukcije (0.25 = 4x usporeno)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--aspect", type=float, default=16 / 9, help="odnos stranica kadra ako original nije nađen")
    args = ap.parse_args()

    kin_suffix = "_kinematics.csv"
    csv_files = sorted(glob.glob(os.path.join(args.input, f"*{kin_suffix}")))
    if not csv_files:
        print(f"Nema *{kin_suffix} fajlova u: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"Pronađeno CSV-ova: {len(csv_files)} (mod: {args.mode})")

    for kin_path in csv_files:
        base = os.path.basename(kin_path)[: -len(kin_suffix)]
        print(f"\nObrada: {base}")
        render_one(kin_path, base, args)

    print("\n=== RENDEROVANJE ČIKICA JE GOTOVO ===")


if __name__ == "__main__":
    main()