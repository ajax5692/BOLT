#!/usr/bin/env python3
"""
BOLT: Behavioral Onset and Locomotion Tracker
Mouse tracking + stimulus-locked speed quantification (freeze / flight)
=======================================================================

Setup assumed (matches your OBS recording):
  * The stimulus monitor preview (grey box with a black sweeping dot) is visible in
    the upper-left corner of the video.
  * The mouse is filmed from above in the arena.

How the stimulus and the mouse are kept separate
------------------------------------------------
1. The stimulus box is a user-defined rectangle (STIM_ROI). It is used ONLY to detect
   when the dot is on screen (count of near-black pixels in that box).
2. The very same rectangle (plus a margin) is MASKED OUT of the mouse tracker, so the
   dot can never be mistaken for the mouse, regardless of pixel brightness.
3. The mouse is tracked only in the remaining pixels.

Mouse tracking
--------------
  * Background = per-pixel high-percentile of brightness over sampled frames
    (the mouse is darker than the bedding, so a bright percentile "removes" it as long as it
    is not parked in one spot for > (100-percentile)% of the video). Alternatively pass
    --background empty_arena.png (a frame without the mouse) which is more robust.
  * Foreground = pixels darker than background by > DARK_THRESH, cleaned by morphology.
  * The largest blob (> MIN_AREA) is the mouse; centroid -> position.
  * If the mouse is not found (e.g. hidden in the tube) the position is NaN and
    is excluded from speed calculations (flagged in the CSV).
  * Position is smoothed (median + Gaussian) before differentiating to speed, which
    suppresses centroid jitter.

Outputs (in --outdir)
---------------------
  <video>_trace.csv      per-frame time, x, y, speed (px/s), motion energy, stimulus flag
  <video>_events.csv     per-stimulus summary (baseline / stimulus / post speeds, peak,
                         latency to peak, % time frozen, freezing & flight indices ...)
  <video>_event<N>.png   speed trace around each stimulus
  <video>_overview.png   whole-session speed trace with stimulus periods shaded
  <video>_tracked.mp4    (optional --save_video) annotated video for visual validation

Usage
-----
  python bolt.py VIDEO.mp4
  python bolt.py VIDEO.mp4 --select_rois          # draw the boxes by mouse
  python bolt.py VIDEO.mp4 --px_per_cm 6.5 --save_video
  python bolt.py VIDEO.mp4 --background empty.png

Requires: opencv-python, numpy, scipy, pandas, matplotlib
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

# ----------------------------------------------------------------------------
# DEFAULT PARAMETERS  (tune for your setup; all can be checked with --save_video)
# ----------------------------------------------------------------------------
# Stimulus preview box (x0, y0, x1, y1) in pixels of the video frame.
# Default fits an 852x480 OBS recording with the grey box in the top-left.
STIM_ROI = (0, 0, 132, 76)
STIM_INNER_PAD = 6            # detection uses the box interior only (skips dark border lines)
STIM_MASK_MARGIN = 6          # extra px masked around STIM_ROI for mouse tracking
# Also mask the text label burned into the video (e.g. "m523"): (x0, y0, x1, y1) or None
LABEL_ROI = (0, 78, 70, 112)

STIM_DARK_LEVEL = 40          # gray level below which a pixel counts as "dot" in the stim box
STIM_MIN_DARK_PX = 30         # >= this many dark px in the box => stimulus on
STIM_MIN_GAP_S = 1.0          # dot-visible gaps shorter than this are merged into one stimulus
STIM_MIN_DURATION_S = 0.5     # ignore stimulus blips shorter than this

# Mouse segmentation
BG_PERCENTILE = 80            # brightness percentile used as background (mouse is darker)
BG_N_FRAMES = 300             # number of frames sampled for the background
DARK_THRESH = 45              # background - frame > this => candidate mouse pixel
MIN_AREA = 250                # min blob area (px) for the mouse (tail is removed by opening)
OPEN_K = 5                    # morphological opening kernel (removes thin tail / noise)
CLOSE_K = 9
BLUR_K = 5

# Speed estimation
SMOOTH_MEDIAN_FRAMES = 5      # median filter on x, y
SMOOTH_GAUSS_SIGMA_S = 0.10   # gaussian smoothing of position, seconds
MAX_GAP_FRAMES = 15           # fill detection gaps up to this long by interpolation

# Event analysis windows (seconds relative to stimulus onset / offset)
BASELINE_S = 5.0
POST_S = 5.0
FREEZE_SPEED_PX_S = 8.0       # speed below this counts as "immobile" (px/s, tune!)
FREEZE_MIN_S = 0.5            # immobile bouts must last at least this long to be "freezing"
FLIGHT_SPEED_PX_S = 150.0     # speed above this counts as "fast / flight-like" (px/s, tune!)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def read_all_gray(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, n, w, h


def select_roi_gui(path, text, default):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    cap.release()
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(text, frame, showCrosshair=False)
    cv2.destroyWindow(text)
    if w == 0 or h == 0:
        return default
    return (int(x), int(y), int(x + w), int(y + h))


def build_background(path, n_frames, percentile):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    stack = np.stack(frames).astype(np.float32)
    return np.percentile(stack, percentile, axis=0).astype(np.float32)


def stimulus_mask(shape, stim_roi, margin, label_roi):
    """255 = tracking allowed, 0 = excluded (stimulus box, label)."""
    m = np.full(shape, 255, np.uint8)
    x0, y0, x1, y1 = stim_roi
    m[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin] = 0
    if label_roi is not None:
        a, b, c, d = label_roi
        m[b:d, a:c] = 0
    return m


def segment_mouse(gray, bg, mask, k_open, k_close):
    """Return (centroid(x,y) or None, area, contour or None, motion-ready fg mask)."""
    g = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0).astype(np.float32)
    diff = bg - g                                   # positive where darker than background
    fg = (diff > DARK_THRESH).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, mask)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)   # kills tail + speckle
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0, None, fg
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None, area, None, fg
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    return (cx, cy), area, c, fg


def find_events(stim_on, fps):
    """Boolean per-frame stimulus flag -> list of (onset_frame, offset_frame)."""
    on = np.asarray(stim_on, bool)
    idx = np.where(on)[0]
    if len(idx) == 0:
        return []
    events = []
    s = p = idx[0]
    gap = int(STIM_MIN_GAP_S * fps)
    for i in idx[1:]:
        if i - p > gap:
            events.append((s, p))
            s = i
        p = i
    events.append((s, p))
    return [(a, b) for a, b in events if (b - a + 1) / fps >= STIM_MIN_DURATION_S]


def interpolate_gaps(arr, max_gap):
    s = pd.Series(arr)
    return s.interpolate(limit=max_gap, limit_area="inside").to_numpy()


def longest_runs(bool_arr, fps, min_s):
    """Return total duration (s) of True-runs lasting >= min_s."""
    tot, run = 0, 0
    for v in list(bool_arr) + [False]:
        if v:
            run += 1
        else:
            if run / fps >= min_s:
                tot += run
            run = 0
    return tot / fps


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def process(args):
    global STIM_ROI, LABEL_ROI
    path = args.video
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(args.outdir, exist_ok=True)

    cap, fps, n_frames, W, H = read_all_gray(path)
    print(f"Video: {W}x{H}, {fps:.2f} fps, {n_frames} frames, {n_frames / fps:.1f} s")

    if args.select_rois:
        STIM_ROI = select_roi_gui(path, "Draw STIMULUS box, press ENTER", STIM_ROI)
        r = select_roi_gui(path, "Draw LABEL text box to ignore (ESC = none)", None)
        LABEL_ROI = r
    print(f"Stimulus ROI: {STIM_ROI}   Label ROI: {LABEL_ROI}")

    # --- background -------------------------------------------------------
    if args.background:
        bg = cv2.cvtColor(cv2.imread(args.background), cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(bg, (BLUR_K, BLUR_K), 0)
    else:
        print("Building background ...")
        bg = cv2.GaussianBlur(build_background(path, BG_N_FRAMES, BG_PERCENTILE),
                              (BLUR_K, BLUR_K), 0)
    if args.arena_roi:
        ax0, ay0, ax1, ay1 = args.arena_roi
        arena = np.zeros((H, W), np.uint8)
        arena[ay0:ay1, ax0:ax1] = 255
    else:
        arena = np.full((H, W), 255, np.uint8)
    track_mask = cv2.bitwise_and(stimulus_mask((H, W), STIM_ROI, STIM_MASK_MARGIN, LABEL_ROI), arena)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(os.path.join(args.outdir, f"{base}_tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # --- frame loop -------------------------------------------------------
    sx0, sy0, sx1, sy1 = STIM_ROI
    xs, ys, areas, stim_dark, motion = [], [], [], [], []
    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # (1) stimulus detection - ONLY inside the stimulus box
        p = STIM_INNER_PAD
        box = gray[sy0 + p:sy1 - p, sx0 + p:sx1 - p]
        stim_dark.append(int((box < STIM_DARK_LEVEL).sum()))

        # (2) mouse tracking - stimulus box & label are masked out
        c, area, cnt, fg = segment_mouse(gray, bg, track_mask, k_open, k_close)
        xs.append(np.nan if c is None else c[0])
        ys.append(np.nan if c is None else c[1])
        areas.append(area)

        # (3) frame-differencing "motion energy" (also masked) - robust freeze readout
        if prev_gray is not None:
            d = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), prev_gray)
            d = cv2.bitwise_and((d > 12).astype(np.uint8) * 255, track_mask)
            motion.append(int(cv2.countNonZero(d)))
        else:
            motion.append(0)
        prev_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if writer is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (sx0, sy0), (sx1, sy1), (0, 165, 255), 1)
            if cnt is not None:
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 1)
                cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
            if stim_dark[-1] >= STIM_MIN_DARK_PX:
                cv2.putText(vis, "STIM", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(vis)
        i += 1
        if i % 1000 == 0:
            print(f"  frame {i}/{n_frames}")
    cap.release()
    if writer is not None:
        writer.release()

    n = len(xs)
    t = np.arange(n) / fps
    x, y = np.array(xs), np.array(ys)
    found = ~np.isnan(x)
    print(f"Mouse detected in {found.mean() * 100:.1f}% of frames")

    # --- position smoothing & speed -------------------------------------
    x_i, y_i = interpolate_gaps(x, MAX_GAP_FRAMES), interpolate_gaps(y, MAX_GAP_FRAMES)
    valid = ~np.isnan(x_i)
    xs_s, ys_s = np.full(n, np.nan), np.full(n, np.nan)

    # smooth each contiguous valid segment separately (so NaN gaps don't smear)
    seg_start = None
    for k in range(n + 1):
        v = k < n and valid[k]
        if v and seg_start is None:
            seg_start = k
        if (not v) and seg_start is not None:
            seg = slice(seg_start, k)
            for src, dst in ((x_i, xs_s), (y_i, ys_s)):
                a = src[seg]
                if len(a) >= SMOOTH_MEDIAN_FRAMES:
                    a = median_filter(a, SMOOTH_MEDIAN_FRAMES, mode="nearest")
                a = gaussian_filter1d(a, SMOOTH_GAUSS_SIGMA_S * fps, mode="nearest")
                dst[seg] = a
            seg_start = None

    scale = 1.0 / args.px_per_cm if args.px_per_cm else 1.0
    unit = "cm/s" if args.px_per_cm else "px/s"
    dx = np.gradient(xs_s) * fps * scale
    dy = np.gradient(ys_s) * fps * scale
    speed = np.hypot(dx, dy)                      # NaN where mouse not tracked
    motion = np.array(motion, float)

    stim_on = np.array(stim_dark) >= STIM_MIN_DARK_PX
    events = find_events(stim_on, fps)
    in_stim = np.zeros(n, bool)
    for a, b in events:
        in_stim[a:b + 1] = True
    print(f"Detected {len(events)} stimulus presentation(s): "
          + ", ".join(f"{a / fps:.1f}-{b / fps:.1f}s" for a, b in events))

    trace = pd.DataFrame({
        "frame": np.arange(n), "time_s": t,
        "x_px": xs_s, "y_px": ys_s, "mouse_detected": found,
        f"speed_{unit.replace('/', '_per_')}": speed,
        "motion_energy_px": motion, "blob_area_px": areas,
        "stim_dark_px": stim_dark, "stimulus_on": in_stim,
    })
    trace.to_csv(os.path.join(args.outdir, f"{base}_trace.csv"), index=False)

    # --- per-event quantification --------------------------------------
    fz_thr = FREEZE_SPEED_PX_S * scale
    fl_thr = FLIGHT_SPEED_PX_S * scale
    rows = []
    for e, (a, b) in enumerate(events, 1):
        pre = slice(max(0, a - int(BASELINE_S * fps)), a)
        dur = slice(a, b + 1)
        post = slice(b + 1, min(n, b + 1 + int(POST_S * fps)))

        def stats(sl):
            s = speed[sl]
            ok = ~np.isnan(s)
            cov = ok.mean() if len(s) else np.nan
            if ok.sum() < 3:
                return dict(mean=np.nan, med=np.nan, peak=np.nan, frozen=np.nan, fast=np.nan,
                            cov=cov, dist=np.nan, motion=np.nanmean(motion[sl]) if len(s) else np.nan,
                            t_peak=np.nan)
            frozen = longest_runs((s < fz_thr)[ok], fps, FREEZE_MIN_S) / (ok.sum() / fps) * 100
            fast = (s[ok] > fl_thr).mean() * 100
            sp = np.where(ok, s, 0)
            return dict(mean=np.nanmean(s), med=np.nanmedian(s), peak=np.nanmax(s),
                        frozen=frozen, fast=fast, cov=cov,
                        dist=sp.sum() / fps, motion=np.mean(motion[sl]),
                        t_peak=(np.nanargmax(s) / fps))

        B, S, P = stats(pre), stats(dur), stats(post)
        # latency (s after onset) of the first frame exceeding flight threshold during stim+post
        win = speed[a:post.stop]
        hit = np.where(win > fl_thr)[0]
        lat_fast = hit[0] / fps if len(hit) else np.nan
        # peak speed latency relative to onset (searching stim + post window)
        if np.any(~np.isnan(win)):
            lat_peak = np.nanargmax(win) / fps
            peak_win = np.nanmax(win)
        else:
            lat_peak, peak_win = np.nan, np.nan
        # latency to freeze: first moment after onset at which the mouse is immobile >= FREEZE_MIN_S
        lat_freeze = np.nan
        w_ok = win.copy()
        run = 0
        for k, v in enumerate(w_ok):
            if not np.isnan(v) and v < fz_thr:
                run += 1
                if run / fps >= FREEZE_MIN_S:
                    lat_freeze = (k - run + 1) / fps
                    break
            else:
                run = 0

        ratio = S["mean"] / B["mean"] if B["mean"] and B["mean"] > 0 else np.nan
        row = {
            "event": e, "onset_s": a / fps, "offset_s": b / fps, "duration_s": (b - a + 1) / fps,
            f"baseline_mean_speed_{unit}": B["mean"], f"stim_mean_speed_{unit}": S["mean"],
            f"post_mean_speed_{unit}": P["mean"],
            f"baseline_peak_speed_{unit}": B["peak"], f"stim_peak_speed_{unit}": S["peak"],
            f"post_peak_speed_{unit}": P["peak"],
            f"peak_speed_stim+post_{unit}": peak_win, "latency_to_peak_s": lat_peak,
            "latency_to_fast_s": lat_fast, "latency_to_freeze_s": lat_freeze,
            "baseline_pct_frozen": B["frozen"], "stim_pct_frozen": S["frozen"],
            "post_pct_frozen": P["frozen"],
            "baseline_pct_fast": B["fast"], "stim_pct_fast": S["fast"], "post_pct_fast": P["fast"],
            "baseline_distance": B["dist"], "stim_distance": S["dist"], "post_distance": P["dist"],
            "baseline_motion_energy": B["motion"], "stim_motion_energy": S["motion"],
            "post_motion_energy": P["motion"],
            "stim_over_baseline_speed_ratio": ratio,
            "tracking_coverage_baseline": B["cov"], "tracking_coverage_stim": S["cov"],
        }
        rows.append(row)

        # per-event plot
        lo, hi = max(0, a - int(BASELINE_S * fps)), post.stop
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        tt = t[lo:hi] - a / fps
        ax[0].plot(tt, speed[lo:hi], lw=1)
        ax[0].axvspan(0, (b - a + 1) / fps, color="orange", alpha=0.3, label="stimulus")
        ax[0].axhline(fz_thr, ls=":", c="b", label="freeze thr")
        ax[0].axhline(fl_thr, ls=":", c="r", label="fast thr")
        ax[0].set_ylabel(f"speed ({unit})")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_title(f"{base} - stimulus {e}")
        ax[1].plot(tt, motion[lo:hi], c="gray", lw=1)
        ax[1].axvspan(0, (b - a + 1) / fps, color="orange", alpha=0.3)
        ax[1].set_ylabel("motion energy (px)")
        ax[1].set_xlabel("time from stimulus onset (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"{base}_event{e}.png"), dpi=130)
        plt.close(fig)

    ev_df = pd.DataFrame(rows)
    ev_df.to_csv(os.path.join(args.outdir, f"{base}_events.csv"), index=False)

    # overview plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, speed, lw=0.6)
    for a, b in events:
        ax.axvspan(a / fps, b / fps, color="orange", alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"speed ({unit})")
    ax.set_title(f"{base} - session overview (orange = stimulus)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{base}_overview.png"), dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    if len(ev_df):
        print("\nPer-event summary:")
        print(ev_df.T.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nResults written to: {os.path.abspath(args.outdir)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--outdir", default="tracking_output")
    ap.add_argument("--px_per_cm", type=float, default=None,
                    help="calibration; if given, speed is reported in cm/s")
    ap.add_argument("--background", default=None, help="image of the empty arena (optional)")
    ap.add_argument("--arena_roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="restrict mouse tracking to this rectangle")
    ap.add_argument("--select_rois", action="store_true", help="draw stimulus/label boxes with the mouse")
    ap.add_argument("--save_video", action="store_true", help="write annotated video for QC")
    process(ap.parse_args())


if __name__ == "__main__":
    main()