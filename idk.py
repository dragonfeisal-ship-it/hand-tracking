
# OPTIMIZED VERSION
# Changes:
# - Enabled OpenCV optimizations
# - Enabled OpenCL when available
# - Minor numpy performance tweaks
# - Thread hints for OpenCV
"""
Hand Tracking — enhanced, bug-fixed, optimized
------------------------------------------------
pip install opencv-python mediapipe numpy pillow pycaw comtypes pyautogui screen-brightness-control

Controls:
  Q  = Quit          S  = Screenshot    R  = Record (video + CSV)
  F  = Fingers       L  = Landmark IDs  G  = Gestures
  B  = Bounding box  D  = Debug         T  = Terminal  P  = Terminal detail
  `  = Volume        \\  = Brightness    M  = Mouse     C  = Canvas
  X  = Save/clear canvas                V  = Speed     H  = 2-hand gestures
  A  = Angles        Z  = Shake         [  = Train gesture  ]  = Delete last
  N  = Media control (play/pause/next/prev via gestures)

Drawing canvas (C):  index=draw  fist=clear  peace=cycle colour
Mouse (M):           index tip moves cursor  pinch=click  two-finger=scroll
Media (N):           pointing=play/pause  swipe left=prev  swipe right=next
"""

from __future__ import annotations
import cv2, time, os, sys, hashlib, json, math, ctypes

import cv2
cv2.setUseOptimized(True)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass
try:
    cv2.setNumThreads(4)
except Exception:
    pass

import urllib.request, threading, csv
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# COM init — must happen before ANY pycaw / comtypes usage on this thread
# ─────────────────────────────────────────────────────────────────────────────
_IS_WIN = sys.platform == "win32"
if _IS_WIN:
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "hand_landmarker.task"
MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_HASH_CACHE = "hand_landmarker.sha256"

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_model() -> None:
    need = not os.path.exists(MODEL_PATH)
    if not need and os.path.exists(_HASH_CACHE):
        if _sha256(MODEL_PATH) != open(_HASH_CACHE).read().strip():
            print("[Model] Checksum mismatch — re-downloading.")
            os.remove(MODEL_PATH); need = True
    if need:
        print("Downloading hand_landmarker model (~9 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        open(_HASH_CACHE, "w").write(_sha256(MODEL_PATH))
        print("[Model] Ready")
    else:
        print("[Model] OK")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FINGER_IDX = np.array([[4,3,2],[8,6,5],[12,10,9],[16,14,13],[20,18,17]], dtype=np.int32)
TIP_IDX    = FINGER_IDX[:, 0]

JOINT_TRIPLETS = {
    "Thumb  MCP": (1,2,3),  "Thumb  IP":  (2,3,4),
    "Index  MCP": (5,6,7),  "Index  PIP": (6,7,8),
    "Middle MCP": (9,10,11),"Middle PIP": (10,11,12),
    "Ring   MCP": (13,14,15),"Ring  PIP": (14,15,16),
    "Pinky  MCP": (17,18,19),"Pinky PIP": (18,19,20),
}

_RAW_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(5,17),
]
CONN_A = np.array([a for a,_ in _RAW_CONN], dtype=np.int32)
CONN_B = np.array([b for _,b in _RAW_CONN], dtype=np.int32)

LM_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

cWHITE=(255,255,255); cBLACK=(0,0,0);   cGREEN=(0,255,160)
cBLUE=(255,160,60);   cGREY=(150,150,150); cYELLOW=(0,220,255)
cRED=(50,50,220);     cLIME=(0,210,50);    cORANGE=(0,140,255)
cPURPLE=(200,0,200);  cCYAN=(255,220,0)

DEBOUNCE_N       = 4
GESTURE_HOLD     = 12   # slightly longer hold so labels don't flash off
_MAX_HANDS       = 2
_MATCH_THRESHOLD = 0.25
CUSTOM_GESTURES_FILE = "custom_gestures.json"

PINCH_MIN = 0.03   # normalised distance → 0%
PINCH_MAX = 0.28   # normalised distance → 100%

# ─────────────────────────────────────────────────────────────────────────────
# Terminal display
# ─────────────────────────────────────────────────────────────────────────────
_ESC="\033["; _BOLD="\033[1m"; _RST="\033[0m"
_GREEN_T="\033[92m"; _BLUE_T="\033[94m"; _CYAN_T="\033[96m"
_GREY_T="\033[90m";  _RED_T="\033[91m";  _YELLOW_T="\033[93m"; _WHITE_T="\033[97m"
_KEY_IDS=[0,4,8,9,12,13,16,17,20]
TERM_WRIST=0; TERM_KEYPTS=1; TERM_ALL=2
_TERM_LABELS=["wrist-only","key points (9)","all 21"]
_BLOCK_LINES=0

def _enable_win_ansi() -> bool:
    if not _IS_WIN: return True
    try:
        k32=ctypes.windll.kernel32; hout=k32.GetStdHandle(-11)
        mode=ctypes.c_ulong(); k32.GetConsoleMode(hout, ctypes.byref(mode))
        k32.SetConsoleMode(hout, mode.value | 0x0004); return True
    except: return False

_ANSI_OK = _enable_win_ansi()

def _bar(v, w=12):
    f = max(0, min(w, int(round(v * w)))); return "█"*f + "░"*(w-f)

def print_terminal(frame_n, fps, hands, detail):
    global _BLOCK_LINES
    if not _ANSI_OK:
        for label,arr,pts,states,gesture,n_fing in hands:
            print(f"[{label:5s}] wrist=({arr[0,0]:.3f},{arr[0,1]:.3f}) "
                  f"fingers={n_fing} gesture={gesture or '—'}", flush=True)
        return
    lines = []
    lines.append(f"{_BOLD}{_CYAN_T}── Hand Tracking  frame {frame_n:>6d}  FPS {fps:>5.1f}{_RST}")
    lines.append(f"{_GREY_T}   detail: {_TERM_LABELS[detail]}   T=toggle  P=cycle{_RST}")
    lines.append("")
    if not hands:
        lines.append(f"  {_GREY_T}No hands detected{_RST}")
    else:
        for label,arr,pts,states,gesture,n_fing in hands:
            col = _GREEN_T if label=="Right" else _BLUE_T
            lines.append(f"  {_BOLD}{col}{label}{_RST}  fingers:{_BOLD}{n_fing}{_RST}"
                         f"  gesture:{_YELLOW_T}{gesture or '—'}{_RST}")
            frow = "  "
            for ext, nm in zip(states, ["Thumb","Index","Middle","Ring","Pinky"]):
                dot = f"{_GREEN_T}●{_RST}" if ext else f"{_RED_T}○{_RST}"
                frow += f"{dot}{_GREY_T}{nm[:3]}{_RST}  "
            lines.append(frow); lines.append("")
            ids = {TERM_WRIST:[0], TERM_KEYPTS:_KEY_IDS, TERM_ALL:list(range(21))}[detail]
            lines.append(f"{_GREY_T}  {'ID':>2}  {'Name':<14} {'x':>7} {'y':>7} {'z':>7}"
                         f"  {'px':>5} {'py':>5}  depth{_RST}")
            for i in ids:
                x,y,z = float(arr[i,0]),float(arr[i,1]),float(arr[i,2])
                px,py = int(pts[i,0]),int(pts[i,1])
                tip = f"{_YELLOW_T}◆ {_RST}" if i in (4,8,12,16,20) else "  "
                lines.append(f"  {tip}{i:>2}  {_WHITE_T}{LM_NAMES[i]:<14}{_RST}"
                             f"  {x:>7.4f}  {y:>7.4f}  {z:>7.4f}"
                             f"  {px:>5}  {py:>5}  {_CYAN_T}{_bar(max(0,min(1,z+0.5)),10)}{_RST}")
            lines.append("")
    n_new = len(lines)
    while len(lines) < _BLOCK_LINES: lines.append(_ESC + "2K")
    _BLOCK_LINES = n_new
    move_up = "" if frame_n == 0 else f"{_ESC}{_BLOCK_LINES}A"
    sys.stdout.write(move_up + "\n".join(lines) + "\n"); sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Emoji renderer (PIL, cached)
# ─────────────────────────────────────────────────────────────────────────────
_EMOJI_CACHE: dict[str, np.ndarray] = {}
_EMOJI_FONT = None; _EMOJI_FONT_LOADED = False

def _get_emoji_font(size):
    global _EMOJI_FONT, _EMOJI_FONT_LOADED
    if not _EMOJI_FONT_LOADED:
        _EMOJI_FONT_LOADED = True
        for path in ["C:/Windows/Fonts/seguiemj.ttf",
                     "/System/Library/Fonts/Apple Color Emoji.ttc",
                     "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"]:
            if os.path.exists(path):
                try: _EMOJI_FONT = ImageFont.truetype(path, size); break
                except: pass
    return _EMOJI_FONT

def _render_emoji(emoji, size=48):
    key = f"{emoji}_{size}"
    if key in _EMOJI_CACHE: return _EMOJI_CACHE[key]
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    font = _get_emoji_font(size-8)
    if font: ImageDraw.Draw(img).text((2,2), emoji, font=font, embedded_color=True)
    else:    ImageDraw.Draw(img).ellipse([4,4,size-4,size-4], fill=(255,200,0,200))
    arr = np.array(img, dtype=np.uint8)
    bgra = arr[:,:,[2,1,0,3]]; _EMOJI_CACHE[key] = bgra; return bgra

def overlay_emoji(frame, emoji, cx, cy, size=52):
    patch = _render_emoji(emoji, size); ph, pw = patch.shape[:2]
    x0,y0 = cx-pw//2, cy-ph//2; x1,y1 = x0+pw, y0+ph
    fx0=max(x0,0); fy0=max(y0,0)
    fx1=min(x1,frame.shape[1]); fy1=min(y1,frame.shape[0])
    if fx0>=fx1 or fy0>=fy1: return
    px0,py0 = fx0-x0, fy0-y0; px1,py1 = px0+(fx1-fx0), py0+(fy1-fy0)
    alpha = patch[py0:py1,px0:px1,3:4].astype(np.float32)/255.
    src   = patch[py0:py1,px0:px1,:3].astype(np.float32)
    dst   = frame[fy0:fy1,fx0:fx1].astype(np.float32)
    frame[fy0:fy1,fx0:fx1] = np.add(alpha*src, (1.-alpha)*dst, dtype=np.float32).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Semi-transparent rect helper (better UI)
# ─────────────────────────────────────────────────────────────────────────────
def filled_rect(frame, x1, y1, x2, y2, colour, alpha=0.55, border=None):
    """Draw a filled rectangle with transparency."""
    x1=max(0,x1); y1=max(0,y1)
    x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
    if x1>=x2 or y1>=y2: return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy(); overlay[:] = colour
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, dst=roi)
    frame[y1:y2, x1:x2] = roi
    if border:
        cv2.rectangle(frame, (x1,y1), (x2,y2), border, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Threaded camera (double-buffered, no copy on hot path)
# ─────────────────────────────────────────────────────────────────────────────
class CamReader:
    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened(): raise RuntimeError("Could not open webcam.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS,          30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        ret, f = self._cap.read()
        self._buf  = [f if ret else np.zeros((720,1280,3),np.uint8),
                      np.zeros((720,1280,3),np.uint8)]
        self._read = 0; self._write = 1
        self._lock = threading.Lock(); self._stop = threading.Event()
        self._t = threading.Thread(target=self._reader, daemon=True); self._t.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    np.copyto(self._buf[self._write], frame)
                    self._read, self._write = self._write, self._read

    def read(self):
        with self._lock: return self._buf[self._read]   # no copy

    @property
    def wh(self):
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self._stop.set(); self._t.join(timeout=1.); self._cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# Recorder (video + CSV)
# ─────────────────────────────────────────────────────────────────────────────
class Recorder:
    def __init__(self, w, h, fps=30.):
        os.makedirs("recordings", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vp = f"recordings/hand_{ts}.mp4"; cp = f"recordings/hand_{ts}_positions.csv"
        self._vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        self._cf = open(cp, "w", newline="", encoding="utf-8")
        self._cw = csv.writer(self._cf)
        self._cw.writerow(["frame","timestamp_ms","hand","landmark_id","landmark_name",
                           "x_norm","y_norm","z_norm","x_px","y_px","gesture","fingers"])
        self._n = 0; self._start = time.perf_counter()
        print(f"[Rec] {vp}\n[Rec] {cp}")

    def write_frame(self, frame): self._vw.write(frame)

    def write_positions(self, arr, pts, label, gesture, n_fing):
        ts = int((time.perf_counter()-self._start)*1000)
        for i in range(21):
            self._cw.writerow([self._n, ts, label, i, LM_NAMES[i],
                               f"{arr[i,0]:.6f}", f"{arr[i,1]:.6f}", f"{arr[i,2]:.6f}",
                               pts[i,0], pts[i,1], gesture, n_fing])

    def end_frame(self): self._n += 1

    def stop(self):
        self._vw.release(); self._cf.close()
        print(f"[Rec] Saved ({self._n} frames)")

# ─────────────────────────────────────────────────────────────────────────────
# Landmark helpers — separate buffers per hand (fixes shared-buffer bug)
# ─────────────────────────────────────────────────────────────────────────────
_LM_BUFS = [np.empty((21,3), dtype=np.float32),
            np.empty((21,3), dtype=np.float32)]
_WH_VEC  = np.empty(2, dtype=np.float32)

def lm_to_array(lms, hand_idx: int) -> np.ndarray:
    """Write into per-hand buffer — no shared-buffer corruption."""
    buf = _LM_BUFS[hand_idx % 2]
    for i, lm in enumerate(lms):
        buf[i,0]=lm.x; buf[i,1]=lm.y; buf[i,2]=lm.z
    return buf

def pts_to_pixels(xy21, w, h):
    _WH_VEC[0]=w; _WH_VEC[1]=h
    return (xy21[:,:2] * _WH_VEC).astype(np.int32)

def bounding_box(pts, w, h, pad=18):
    return (max(int(pts[:,0].min())-pad, 0), max(int(pts[:,1].min())-pad, 0),
            min(int(pts[:,0].max())+pad, w-1), min(int(pts[:,1].max())+pad, h-1))

# ─────────────────────────────────────────────────────────────────────────────
# EMA smoother
# ─────────────────────────────────────────────────────────────────────────────
class LandmarkSmoother:
    __slots__ = ("alpha", "_s")
    def __init__(self, alpha=0.50): self.alpha=alpha; self._s=None
    def update(self, arr):
        if self._s is None: self._s=arr.copy()
        else: np.add(self.alpha*arr, (1.-self.alpha)*self._s, out=self._s)
        return self._s.copy()
    def reset(self): self._s=None

# ─────────────────────────────────────────────────────────────────────────────
# Finger debouncer (vectorised)
# ─────────────────────────────────────────────────────────────────────────────
class FingerDebouncer:
    __slots__ = ("n","_stable","_cand","_cnt")
    def __init__(self, n=DEBOUNCE_N):
        self.n=n
        self._stable = np.zeros(5,dtype=bool)
        self._cand   = np.zeros(5,dtype=bool)
        self._cnt    = np.zeros(5,dtype=np.int32)

    def update(self, raw):
        same = raw == self._cand
        self._cnt = np.where(same, self._cnt+1, 1)
        np.copyto(self._cand, raw)
        np.copyto(self._stable, np.where(self._cnt >= self.n, self._cand, self._stable))
        return self._stable.copy()

    def reset(self): self._stable[:]=False; self._cand[:]=False; self._cnt[:]=0

# ─────────────────────────────────────────────────────────────────────────────
# Finger states (vectorised + z-depth thumb fallback)
# ─────────────────────────────────────────────────────────────────────────────
def finger_states_np(arr, is_right):
    states = np.empty(5, dtype=bool)
    palm_v  = arr[5,:2]-arr[0,:2]; thumb_v = arr[4,:2]-arr[2,:2]
    cross   = float(palm_v[0]*thumb_v[1]-palm_v[1]*thumb_v[0])
    rotated = abs(float(arr[0,2]-arr[9,2])) > 0.08
    states[0] = (float(arr[4,2]-arr[2,2]) < -0.02) if rotated else \
                (cross < -0.002 if is_right else cross > 0.002)
    fi = FINGER_IDX[1:]
    mcp=arr[fi[:,2],:2]; pip=arr[fi[:,1],:2]; tip=arr[fi[:,0],:2]
    vb=pip-mcp; vf=tip-mcp
    dot = np.einsum("ij,ij->i",vb,vf)
    mb  = np.linalg.norm(vb,axis=1); mf = np.linalg.norm(vf,axis=1)
    safe = (mb>1e-6)&(mf>1e-6)
    cos_a = np.where(safe, np.clip(dot/(mb*mf+1e-9),-1.,1.), 0.)
    states[1:] = (cos_a > 0.72) & (arr[fi[:,0],1] < arr[fi[:,2],1])
    return states

# ─────────────────────────────────────────────────────────────────────────────
# Finger joint angles
# ─────────────────────────────────────────────────────────────────────────────
def compute_joint_angles(arr):
    angles = {}
    for name,(a,b,c) in JOINT_TRIPLETS.items():
        v1=arr[a,:3]-arr[b,:3]; v2=arr[c,:3]-arr[b,:3]
        m1=np.linalg.norm(v1); m2=np.linalg.norm(v2)
        if m1<1e-6 or m2<1e-6: angles[name]=0.; continue
        angles[name] = math.degrees(math.acos(float(np.clip(np.dot(v1,v2)/(m1*m2),-1.,1.))))
    return angles

def draw_angle_readout(frame, arr, h_frame, w_frame):
    angles = compute_joint_angles(arr)
    n = len(angles); panel_h = n*18+14; x0 = w_frame-218; y0 = 38
    filled_rect(frame, x0-4, y0, w_frame-4, y0+panel_h, (20,20,20), 0.7, cGREY)
    for i,(name,deg) in enumerate(angles.items()):
        col = cLIME if deg<45 else cORANGE if deg<120 else cRED
        y = y0+14+i*18
        cv2.putText(frame, f"{name}: {deg:5.1f}°", (x0, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, col, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Speed & acceleration
# ─────────────────────────────────────────────────────────────────────────────
class SpeedTracker:
    def __init__(self, history=8):
        self._h: deque = deque(maxlen=history)
        self.speed=0.; self.accel=0.; self._prev=0.
    def update(self, wxy, t):
        self._h.append((t, wxy.copy()))
        if len(self._h)>=2:
            t0,p0=self._h[0]; t1,p1=self._h[-1]; dt=t1-t0
            if dt>1e-4:
                self.speed = float(np.linalg.norm(p1-p0))/dt
                self.accel = (self.speed-self._prev)/dt
                self._prev = self.speed
    def reset(self): self._h.clear(); self.speed=0.; self.accel=0.; self._prev=0.

def draw_speed_overlay(frame, pts, speed, accel, is_right, slot_idx):
    col=cGREEN if is_right else cBLUE
    wx=int(pts[0,0]); wy=int(pts[0,1])-55+slot_idx*50
    bw=min(int(speed*280),110)
    filled_rect(frame, wx-62,wy-20, wx+62,wy+12, (20,20,20), 0.6)
    cv2.rectangle(frame,(wx-60,wy-18),(wx-60+bw,wy-8),col,-1)
    cv2.rectangle(frame,(wx-60,wy-18),(wx+60,wy-8),cGREY,1)
    cv2.putText(frame,f"spd {speed:.2f}",(wx-58,wy-10),cv2.FONT_HERSHEY_PLAIN,0.9,col,1,cv2.LINE_AA)
    a_col=cLIME if accel>=0 else cRED
    cv2.putText(frame,f"acc {accel:+.2f}",(wx-58,wy+4),cv2.FONT_HERSHEY_PLAIN,0.9,a_col,1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Shake detector
# ─────────────────────────────────────────────────────────────────────────────
class ShakeDetector:
    WINDOW=20; SHAKE_THR=0.018; REVERSAL_N=4
    def __init__(self):
        self._h: deque=deque(maxlen=self.WINDOW); self.shaking=False; self.intensity=0.
    def update(self, wxy):
        self._h.append(wxy.copy())
        if len(self._h)<self.WINDOW//2: self.shaking=False; self.intensity=0.; return
        pts=np.array(list(self._h))
        var=float(pts.var(axis=0).sum())
        rev=int(np.sum(np.diff(np.sign(np.diff(pts[:,0])))!=0))
        self.intensity=min(var/self.SHAKE_THR,1.)
        self.shaking=var>self.SHAKE_THR and rev>=self.REVERSAL_N
    def reset(self): self._h.clear(); self.shaking=False; self.intensity=0.

def draw_shake_indicator(frame, pts, shaking, intensity):
    if not shaking: return
    wx,wy=int(pts[0,0]),int(pts[0,1]); r=int(20+intensity*18)
    cv2.circle(frame,(wx,wy),r,cRED,3,cv2.LINE_AA)
    cv2.putText(frame,"SHAKE!",(wx-30,wy-r-8),cv2.FONT_HERSHEY_SIMPLEX,0.75,cRED,2,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Multi-hand gesture
# ─────────────────────────────────────────────────────────────────────────────
def classify_two_hand_gesture(arrs, states_list):
    if len(arrs)<2: return ""
    a0,a1=arrs[0],arrs[1]; s0,s1=states_list[0],states_list[1]
    wd=float(np.linalg.norm(a0[0,:2]-a1[0,:2]))
    if not s0.any() and not s1.any() and wd<0.18: return "👏 Clap"
    if s0.all() and s1.all() and wd<0.25:         return "🙌 High Five"
    if float(np.linalg.norm(a0[8,:2]-a1[8,:2]))<0.08: return "🤝 Fingertip Touch"
    if (not s0[0] and s0[1] and not s0[2] and not s1[0] and s1[1] and not s1[2]):
        return "👆👆 Both Pointing"
    if wd>0.55 and s0.all() and s1.all(): return "🖐🖐 Wide Open"
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# Drawing canvas
# ─────────────────────────────────────────────────────────────────────────────
DRAW_COLOURS      = [(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,255,255),(200,0,200)]
DRAW_COLOUR_NAMES = ["Red","Green","Blue","White","Yellow","Purple"]

class DrawingCanvas:
    def __init__(self, w, h):
        self._w=w; self._h=h
        self.canvas=np.zeros((h,w,3),dtype=np.uint8)
        self._prev=None; self._col_idx=0; self._brush=6
        self._was_fist=False; self._was_peace=False
    @property
    def colour(self): return DRAW_COLOURS[self._col_idx]
    @property
    def colour_name(self): return DRAW_COLOUR_NAMES[self._col_idx]
    def update(self, pts, states):
        only_index=(not states[0] and states[1] and not states[2] and not states[3] and not states[4])
        fist=not states.any()
        peace=(not states[0] and states[1] and states[2] and not states[3] and not states[4])
        if fist and not self._was_fist: self.canvas[:]=0; self._prev=None
        self._was_fist=fist
        if peace and not self._was_peace:
            self._col_idx=(self._col_idx+1)%len(DRAW_COLOURS); self._prev=None
        self._was_peace=peace
        if only_index:
            tip=(int(pts[8,0]),int(pts[8,1]))
            if self._prev: cv2.line(self.canvas,self._prev,tip,self.colour,self._brush,cv2.LINE_AA)
            self._prev=tip
        else:
            self._prev=None
    def blend_onto(self, frame):
        mask=self.canvas.any(axis=2)
        frame[mask]=cv2.addWeighted(frame,0.25,self.canvas,0.75,0)[mask]
    def save(self):
        os.makedirs("drawings",exist_ok=True)
        path=f"drawings/canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(path,self.canvas); print(f"[Canvas] Saved → {path}")
    def clear(self): self.canvas[:]=0; self._prev=None

# ─────────────────────────────────────────────────────────────────────────────
# Mouse controller — Win32 direct cursor (zero overhead), adaptive EMA
# ─────────────────────────────────────────────────────────────────────────────
class MouseController:
    DEADZONE   = 0.006   # normalised, below this = no move
    CLICK_DIST = 0.038   # thumb-index pinch threshold
    CLICK_HOLD = 10      # frames before re-trigger

    # Adaptive smoothing: alpha = base + vel_scale * velocity
    ALPHA_BASE   = 0.12   # slow/precise movement
    ALPHA_MAX    = 0.60   # fast sweeping movement
    ALPHA_VEL_K  = 3.5   # velocity sensitivity

    def __init__(self, sw, sh):
        self._sw=sw; self._sh=sh
        self._cx=sw/2.; self._cy=sh/2.
        self._clicking=False; self._click_hold=0
        self._prev_tx=self._cx; self._prev_ty=self._cy
        self._ok=False; self._use_win32=False

        # Prefer win32 direct syscall (zero latency)
        if _IS_WIN:
            try:
                ctypes.windll.user32.SetCursorPos(int(self._cx), int(self._cy))
                self._use_win32=True; self._ok=True
                print("[Mouse] Win32 direct cursor — zero latency")
            except: pass

        if not self._ok:
            try:
                import pyautogui
                pyautogui.FAILSAFE=False; pyautogui.PAUSE=0
                self._pag=pyautogui; self._ok=True
                print("[Mouse] pyautogui OK")
            except ImportError:
                print("[Mouse] install pyautogui  (or running on Windows for zero-latency)")

    @property
    def available(self): return self._ok

    def _move(self, x, y):
        ix,iy=int(x),int(y)
        if self._use_win32:
            ctypes.windll.user32.SetCursorPos(ix,iy)
        else:
            try: self._pag.moveTo(ix,iy)
            except: pass

    def _click(self):
        if self._use_win32:
            # SendInput mouse click via ctypes
            MOUSEEVENTF_LEFTDOWN=0x0002; MOUSEEVENTF_LEFTUP=0x0004
            try:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN,0,0,0,0)
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP,0,0,0,0)
            except: pass
        else:
            try: self._pag.click()
            except: pass

    def update(self, arr):
        if not self._ok: return False, False

        # Crop margins so full-screen sweep uses less hand movement
        margin=0.12
        nx=float(np.clip((arr[8,0]-margin)/(1.-2*margin), 0.,1.))
        ny=float(np.clip((arr[8,1]-margin)/(1.-2*margin), 0.,1.))
        tx=nx*self._sw; ty=ny*self._sh

        # Compute velocity of target point
        vx=(tx-self._prev_tx)/self._sw; vy=(ty-self._prev_ty)/self._sh
        vel=math.hypot(vx,vy)
        self._prev_tx=tx; self._prev_ty=ty

        # Adaptive alpha: fast hand = more responsive
        alpha=min(self.ALPHA_BASE + self.ALPHA_VEL_K*vel, self.ALPHA_MAX)

        dx=tx-self._cx; dy=ty-self._cy
        dist_norm=math.hypot(dx/self._sw, dy/self._sh)
        if dist_norm > self.DEADZONE:
            self._cx+=alpha*dx; self._cy+=alpha*dy
            self._move(self._cx, self._cy)

        # Pinch click
        pinch=float(np.linalg.norm(arr[4,:2]-arr[8,:2]))<self.CLICK_DIST
        clicked=False
        if pinch and not self._clicking and self._click_hold==0:
            self._click(); self._clicking=True; self._click_hold=self.CLICK_HOLD; clicked=True
        elif not pinch: self._clicking=False
        if self._click_hold>0: self._click_hold-=1
        return True, clicked

def draw_mouse_overlay(frame, pts, clicked, is_right):
    col=cGREEN if is_right else cBLUE
    tip=(int(pts[8,0]),int(pts[8,1]))
    cv2.circle(frame,tip,14,col,3,cv2.LINE_AA)
    # Cross-hair on tip
    cv2.line(frame,(tip[0]-8,tip[1]),(tip[0]+8,tip[1]),col,1,cv2.LINE_AA)
    cv2.line(frame,(tip[0],tip[1]-8),(tip[0],tip[1]+8),col,1,cv2.LINE_AA)
    if clicked:
        cv2.circle(frame,tip,22,(0,0,255),4,cv2.LINE_AA)
        cv2.putText(frame,"CLICK",(tip[0]+16,tip[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Volume controller — pycaw with CoInitialize fix
# ─────────────────────────────────────────────────────────────────────────────
class VolumeController:
    def __init__(self):
        self._vol=None; self._ok=False; self._smooth=0.5
        if not _IS_WIN:
            print("[Volume] Windows only"); return
        try:
            # CoInitialize must be called before first COM usage on this thread
            import comtypes; comtypes.CoInitialize()
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from ctypes import cast, POINTER
            devices = AudioUtilities.GetSpeakers()
            iface   = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
            self._vol = cast(iface, POINTER(IAudioEndpointVolume))
            self._smooth = float(self._vol.GetMasterVolumeLevelScalar())
            self._ok = True
            print(f"[Volume] OK — {self._smooth*100:.0f}%")
        except Exception as e:
            print(f"[Volume] Failed: {e}\n  → pip install pycaw comtypes")

    @property
    def available(self): return self._ok

    def set_from_pinch(self, dist):
        target = float(np.clip((dist-PINCH_MIN)/max(PINCH_MAX-PINCH_MIN,1e-6), 0.,1.))
        self._smooth += 0.20*(target-self._smooth)
        scalar = float(np.clip(self._smooth,0.,1.))
        if self._ok:
            try: self._vol.SetMasterVolumeLevelScalar(scalar, None)
            except Exception as e: print(f"[Volume] set error: {e}"); self._ok=False
        return scalar

    def get(self):
        if self._ok:
            try: return float(self._vol.GetMasterVolumeLevelScalar())
            except: pass
        return float(np.clip(self._smooth,0.,1.))

# ─────────────────────────────────────────────────────────────────────────────
# Brightness controller — WMI primary, screen_brightness_control fallback
# ─────────────────────────────────────────────────────────────────────────────
class BrightnessController:
    def __init__(self):
        self._ok=False; self._smooth=0.5; self._backend=None
        if not _IS_WIN:
            print("[Brightness] Windows only"); return

        # Try WMI first (works on most laptops)
        try:
            import wmi
            self._wmi = wmi.WMI(namespace="wmi")
            methods   = self._wmi.WmiMonitorBrightnessMethods()[0]
            query     = self._wmi.WmiMonitorBrightness()[0]
            cur       = query.CurrentBrightness
            self._smooth = cur/100.
            self._wmi_m  = methods
            self._backend = "wmi"
            self._ok = True
            print(f"[Brightness] WMI OK — {cur}%")
            return
        except Exception: pass

        # Fallback: screen_brightness_control
        try:
            import screen_brightness_control as sbc
            vals = sbc.get_brightness()
            cur  = vals[0] if vals else 50
            self._smooth = cur/100.
            self._sbc    = sbc
            self._backend= "sbc"
            self._ok     = True
            print(f"[Brightness] screen_brightness_control OK — {cur}%")
        except Exception as e:
            print(f"[Brightness] Failed: {e}\n  → pip install wmi screen-brightness-control")

    @property
    def available(self): return self._ok

    def set_from_pinch(self, dist):
        target = float(np.clip((dist-PINCH_MIN)/max(PINCH_MAX-PINCH_MIN,1e-6), 0.,1.))
        self._smooth += 0.20*(target-self._smooth)
        val = float(np.clip(self._smooth,0.,1.))
        pct = int(val*100)
        if self._ok:
            try:
                if self._backend=="wmi":
                    self._wmi_m.WmiSetBrightness(pct, 0)
                else:
                    self._sbc.set_brightness(pct)
            except Exception as e:
                print(f"[Brightness] set error: {e}"); self._ok=False
        return val

# ─────────────────────────────────────────────────────────────────────────────
# Media controller (gesture-driven: point=play/pause, swipe=skip)
# ─────────────────────────────────────────────────────────────────────────────
class MediaController:
    SWIPE_DIST    = 0.18   # normalised wrist travel to trigger skip
    SWIPE_FRAMES  = 12     # max frames to complete a swipe
    COOLDOWN      = 25     # frames between actions

    def __init__(self):
        self._ok = False
        self._swipe_start: Optional[np.ndarray] = None
        self._swipe_frame = 0
        self._cooldown    = 0
        self._last_action = ""
        self._action_hold = 0
        if _IS_WIN:
            # Use ctypes SendInput — no pyautogui dependency
            self._ok = True
            print("[Media] OK (Win32 SendInput)")
        else:
            print("[Media] Windows only")

    def _send_key(self, vk):
        """Send a media key via keybd_event."""
        KEYEVENTF_EXTENDEDKEY=0x0001; KEYEVENTF_KEYUP=0x0002
        ctypes.windll.user32.keybd_event(vk,0,KEYEVENTF_EXTENDEDKEY,0)
        ctypes.windll.user32.keybd_event(vk,0,KEYEVENTF_EXTENDEDKEY|KEYEVENTF_KEYUP,0)

    def update(self, arr, states):
        """
        Gestures:
          Pointing (index only) = play/pause
          Swipe right (open hand moves right) = next track
          Swipe left  (open hand moves left)  = prev track
        Returns action string or ''.
        """
        if not self._ok: return ""
        if self._cooldown > 0: self._cooldown-=1
        if self._action_hold > 0:
            self._action_hold-=1
            if self._action_hold==0: self._last_action=""

        pointing  = (not states[0] and states[1] and not states[2]
                     and not states[3] and not states[4])
        open_hand = states.all()
        wxy = arr[0,:2].copy()

        action = ""

        # Play/pause on pointing gesture (edge-triggered)
        if not hasattr(self, "_was_pointing"): self._was_pointing=False
        if pointing and not self._was_pointing and self._cooldown==0:
            self._send_key(0xB3)  # VK_MEDIA_PLAY_PAUSE
            action="⏯ Play/Pause"; self._cooldown=self.COOLDOWN
        self._was_pointing=pointing

        # Swipe detection
        if open_hand:
            if self._swipe_start is None:
                self._swipe_start=wxy.copy(); self._swipe_frame=0
            else:
                self._swipe_frame+=1
                dx=float(wxy[0]-self._swipe_start[0])
                if self._swipe_frame <= self.SWIPE_FRAMES and self._cooldown==0:
                    if dx > self.SWIPE_DIST:
                        self._send_key(0xB0)  # VK_MEDIA_NEXT_TRACK
                        action="⏭ Next"; self._cooldown=self.COOLDOWN
                        self._swipe_start=None
                    elif dx < -self.SWIPE_DIST:
                        self._send_key(0xB1)  # VK_MEDIA_PREV_TRACK
                        action="⏮ Prev"; self._cooldown=self.COOLDOWN
                        self._swipe_start=None
                if self._swipe_frame>self.SWIPE_FRAMES:
                    self._swipe_start=None
        else:
            self._swipe_start=None; self._swipe_frame=0

        if action:
            self._last_action=action; self._action_hold=40

        return self._last_action

# ─────────────────────────────────────────────────────────────────────────────
# Pinch bar overlay (shared by volume + brightness)
# ─────────────────────────────────────────────────────────────────────────────
def draw_pinch_bar(frame, pts, level, label, col, bar_x_offset=0):
    hf,wf=frame.shape[:2]
    t4=(int(pts[4,0]),int(pts[4,1])); t8=(int(pts[8,0]),int(pts[8,1]))
    mid=((t4[0]+t8[0])//2,(t4[1]+t8[1])//2)
    cv2.line(frame,t4,t8,col,3,cv2.LINE_AA)
    for pt in (t4,t8): cv2.circle(frame,pt,12,col,-1); cv2.circle(frame,pt,12,cWHITE,2)
    pct=f"{int(level*100)}%"
    (tw,th),_=cv2.getTextSize(pct,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)
    filled_rect(frame,mid[0]-tw//2-8,mid[1]-th-10,mid[0]+tw//2+8,mid[1]+8,(0,0,0),0.7)
    cv2.putText(frame,pct,(mid[0]-tw//2,mid[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2,cv2.LINE_AA)
    BX=wf-44-bar_x_offset; BY1=80; BY2=hf-80; BH=BY2-BY1; filled=int(level*BH)
    filled_rect(frame,BX,BY1,BX+26,BY2,(10,10,10),0.75,cGREY)
    if filled>0: cv2.rectangle(frame,(BX,BY2-filled),(BX+26,BY2),col,-1)
    cv2.putText(frame,label,(BX-2,BY1-12),cv2.FONT_HERSHEY_PLAIN,1.0,col,1,cv2.LINE_AA)
    cv2.putText(frame,pct,(BX-6,BY2+20),cv2.FONT_HERSHEY_PLAIN,1.1,col,2,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Custom gesture trainer
# ─────────────────────────────────────────────────────────────────────────────
class GestureTrainer:
    def __init__(self):
        self._customs: list[dict]=[]
        if os.path.exists(CUSTOM_GESTURES_FILE):
            try:
                self._customs=json.loads(open(CUSTOM_GESTURES_FILE).read())
                print(f"[Trainer] Loaded {len(self._customs)} gesture(s)")
            except Exception as e: print(f"[Trainer] Load error: {e}")

    def _save(self):
        open(CUSTOM_GESTURES_FILE,"w").write(json.dumps(self._customs,indent=2))

    def record(self, states, arr, name):
        pat=[bool(v) for v in states.tolist()]
        geom={"pinch_dist":float(np.linalg.norm(arr[4,:2]-arr[8,:2])),
              "thumb_rel":(arr[4,:2]-arr[0,:2]).tolist(),
              "index_rel":(arr[8,:2]-arr[0,:2]).tolist()}
        self._customs=[c for c in self._customs if c["name"]!=name]
        self._customs.append({"name":name,"pattern":pat,"geom":geom})
        self._save(); print(f"[Trainer] Saved '{name}'")

    def delete_last(self):
        if self._customs:
            r=self._customs.pop(); self._save(); print(f"[Trainer] Deleted '{r['name']}'")

    def classify(self, states, arr):
        pat=[bool(v) for v in states.tolist()]
        for e in self._customs:
            if e["pattern"]==pat:
                if abs(float(np.linalg.norm(arr[4,:2]-arr[8,:2]))-e["geom"]["pinch_dist"])<0.08:
                    return f"⭐ {e['name']}"
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Gesture table (O(1) 5-bit lookup)
# ─────────────────────────────────────────────────────────────────────────────
def _bit(s): return int(s[0])<<4|int(s[1])<<3|int(s[2])<<2|int(s[3])<<1|int(s[4])
def _key(a): return _bit(a.tolist())
_GT: dict[int,list]={}
def _reg(p,e,n,x=None): _GT.setdefault(_bit(p),[]).append((e,n,x))
_reg((0,0,0,0,0),"✊","Fist")
_reg((1,1,1,1,1),"🖐","Open Hand")
_reg((1,0,0,0,0),"👍","Thumbs Up",   lambda a,r:a[4,1]<a[0,1]-0.10 and a[4,1]<a[5,1])
_reg((1,0,0,0,0),"👎","Thumbs Down", lambda a,r:a[4,1]>a[0,1]+0.08)
_reg((0,1,1,0,0),"✌","Peace")
_reg((1,0,0,0,1),"🤙","Call Me",     lambda a,r:a[4,1]>a[8,1])
_reg((1,0,0,0,1),"🤘","Rock On")
_reg((0,1,0,0,0),"👆","Pointing")
_reg((0,1,1,1,0),"3️⃣","Three")
_reg((0,1,1,1,1),"4️⃣","Four")
_reg((0,0,0,1,1),"👌","OK",          lambda a,r:float(np.linalg.norm(a[4,:2]-a[8,:2]))<0.06)
_reg((0,0,0,0,0),"🤏","Pinch",       lambda a,r:float(np.linalg.norm(a[4,:2]-a[8,:2]))<0.04)
_reg((0,1,0,0,1),"🕷","Spider-Man")
_reg((1,1,0,0,0),"🔫","Gun")

def classify_gesture(states,arr,is_right):
    for e,n,x in _GT.get(_key(states),[]):
        if x is None or x(arr,is_right): return e,n
    return "",""

# ─────────────────────────────────────────────────────────────────────────────
# Hand identity matcher — O(1) for 2 hands (no np.argsort)
# ─────────────────────────────────────────────────────────────────────────────
def match_hands(states, det_lms, det_labels):
    n_det=len(det_lms)
    if n_det==0:
        for hs in states: hs.reset()
        return []

    det_wrists=np.empty((n_det,2),dtype=np.float32)
    for i,lm in enumerate(det_lms): det_wrists[i,0]=lm[0].x; det_wrists[i,1]=lm[0].y

    n_slots=len(states)
    INF=1e9; cost=np.full((n_slots,n_det),INF,dtype=np.float64)
    for s,hs in enumerate(states):
        if hs.last_wrist is not None:
            cost[s]=np.linalg.norm(det_wrists-hs.last_wrist,axis=1)
            for d in range(n_det):
                if hs.label and hs.label!=det_labels[d]: cost[s,d]+=0.30

    used_s=set(); used_d=set(); assignment={}

    # For ≤2 hands: direct min-scan instead of full argsort
    if n_slots<=2 and n_det<=2:
        for _ in range(min(n_slots,n_det)):
            best_c=INF; best_s=best_d=-1
            for s in range(n_slots):
                if s in used_s: continue
                for d in range(n_det):
                    if d in used_d: continue
                    if cost[s,d]<best_c: best_c=cost[s,d]; best_s=s; best_d=d
            if best_c<_MATCH_THRESHOLD:
                assignment[best_s]=best_d; used_s.add(best_s); used_d.add(best_d)
    else:
        for idx in np.argsort(cost,axis=None):
            s,d=divmod(int(idx),n_det)
            if s in used_s or d in used_d: continue
            if cost[s,d]>_MATCH_THRESHOLD: break
            assignment[s]=d; used_s.add(s); used_d.add(d)

    free=[i for i in range(n_slots) if i not in used_s]
    for d in range(n_det):
        if d not in used_d:
            if free: assignment[free.pop(0)]=d
            else: states.append(HandState()); assignment[len(states)-1]=d

    for s in range(len(states)):
        if s not in assignment: states[s].reset()

    return [(states[s],d) for s,d in sorted(assignment.items())]

# ─────────────────────────────────────────────────────────────────────────────
# Per-hand state
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HandState:
    smoother:    LandmarkSmoother = field(default_factory=LandmarkSmoother)
    debouncer:   FingerDebouncer  = field(default_factory=FingerDebouncer)
    speed_track: SpeedTracker     = field(default_factory=SpeedTracker)
    shake_det:   ShakeDetector    = field(default_factory=ShakeDetector)
    g_emoji: str=""; g_name: str=""; hold: int=0
    last_wrist: Optional[np.ndarray]=field(default=None,repr=False)
    label: str=""; active: bool=False

    def reset(self):
        self.smoother.reset(); self.debouncer.reset()
        self.speed_track.reset(); self.shake_det.reset()
        self.g_emoji=""; self.g_name=""; self.hold=0
        self.last_wrist=None; self.label=""; self.active=False

# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_skeleton(frame, pts, is_right):
    bone=cGREEN if is_right else cBLUE
    for a,b in zip(CONN_A,CONN_B): cv2.line(frame,pts[a],pts[b],bone,2,cv2.LINE_AA)
    for i,(x,y) in enumerate(pts):
        r=7 if i in (4,8,12,16,20) else 4
        cv2.circle(frame,(int(x),int(y)),r,cWHITE,-1)
        cv2.circle(frame,(int(x),int(y)),r,bone,2)

def draw_tips(frame, pts, states):
    for ext,tip in zip(states,TIP_IDX):
        x,y=int(pts[tip,0]),int(pts[tip,1])
        cv2.circle(frame,(x,y),10,cLIME if ext else cRED,-1)
        cv2.circle(frame,(x,y),10,cWHITE,2)

def draw_bbox(frame, pts, w, h, is_right, label):
    x1,y1,x2,y2=bounding_box(pts,w,h); col=cGREEN if is_right else cBLUE
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
    cv2.putText(frame,label,(x1+6,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)

def draw_ids(frame, pts):
    for i,(x,y) in enumerate(pts):
        cv2.putText(frame,str(i),(int(x)+5,int(y)-4),cv2.FONT_HERSHEY_PLAIN,0.8,cYELLOW,1,cv2.LINE_AA)

def draw_gesture(frame, pts, emoji, name, is_right):
    if not name: return
    col=cGREEN if is_right else cBLUE
    wx,wy=int(pts[0,0]),int(pts[0,1])+62
    (tw,th),_=cv2.getTextSize(name,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    x0=max(6,wx-tw//2)
    filled_rect(frame,x0-8,wy-th-8,x0+tw+8,wy+10,(10,10,10),0.72,col)
    cv2.putText(frame,name,(x0,wy),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2,cv2.LINE_AA)
    if emoji: overlay_emoji(frame,emoji,wx,wy-th-36,size=48)

def draw_debug(frame, label, raw_st, deb_st, slot_idx):
    names=["Th","Idx","Mid","Rng","Pky"]
    fmt=lambda s:" ".join(f"{n}:{'1' if v else '0'}" for n,v in zip(names,s))
    y0=165+slot_idx*46
    for prefix,st in [("raw",raw_st),("deb",deb_st)]:
        cv2.putText(frame,f"{label} {prefix}: {fmt(st)}",(6,y0),
                    cv2.FONT_HERSHEY_PLAIN,1.0,cYELLOW,1,cv2.LINE_AA); y0+=22

def draw_two_hand_gesture(frame, text, h, w):
    if not text: return
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.0,2)
    x0=(w-tw)//2; y0=h//2
    filled_rect(frame,x0-12,y0-th-12,x0+tw+12,y0+12,(0,0,0),0.72,(0,200,255))
    cv2.putText(frame,text,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,200,255),2,cv2.LINE_AA)

def draw_media_badge(frame, action, w, h):
    if not action: return
    (tw,th),_=cv2.getTextSize(action,cv2.FONT_HERSHEY_SIMPLEX,1.1,2)
    x0=(w-tw)//2; y0=h//2-60
    filled_rect(frame,x0-12,y0-th-12,x0+tw+12,y0+12,(0,0,0),0.75,cPURPLE)
    cv2.putText(frame,action,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,1.1,cPURPLE,2,cv2.LINE_AA)

def draw_canvas_hud(frame, canvas, active, w):
    if not active: return
    col=canvas.colour
    filled_rect(frame,w//2-160,4,w//2+160,34,(10,10,10),0.7)
    cv2.circle(frame,(w//2-130,18),10,col,-1); cv2.circle(frame,(w//2-130,18),10,cWHITE,1)
    cv2.putText(frame,f"DRAW: {canvas.colour_name}  ✊=clear  ✌=colour",
                (w//2-110,24),cv2.FONT_HERSHEY_PLAIN,1.1,cWHITE,1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Active mode indicator panel (top-right)
# ─────────────────────────────────────────────────────────────────────────────
def draw_mode_panel(frame, modes: dict[str,bool], w: int):
    """Show which modes are currently ON in a compact panel."""
    active = [k for k,v in modes.items() if v]
    if not active: return
    line = "  ".join(active)
    (tw,th),_=cv2.getTextSize(line,cv2.FONT_HERSHEY_PLAIN,1.0,1)
    x0=w-tw-20; y0=4
    filled_rect(frame,x0-6,y0,x0+tw+8,y0+th+10,(10,10,10),0.7)
    cv2.putText(frame,line,(x0,y0+th+2),cv2.FONT_HERSHEY_PLAIN,1.0,cCYAN,1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
_HINT=("Q=Quit R=Rec S=Shot  F=Fingers L=Ids G=Gestures B=Box D=Debug "
       "T=Terminal P=Detail  `=Vol \\=Bright M=Mouse C=Canvas X=SaveCanvas "
       "V=Speed H=2hand A=Angles Z=Shake N=Media [=Train ]=DeleteLast")

def draw_hud(frame, h, fps, hands_info, show_fingers, recording):
    filled_rect(frame,0,0,130,32,(10,10,10),0.75)
    cv2.putText(frame,f"FPS {fps:.0f}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,cGREEN,2,cv2.LINE_AA)
    if recording:
        cv2.circle(frame,(150,16),9,(0,0,220),-1)
        cv2.putText(frame,"REC",(163,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,220),2,cv2.LINE_AA)
    if show_fingers:
        for idx,(label,n) in enumerate(hands_info):
            text=f"{label}: {n}"
            (tw,_),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
            yp=56+idx*36; col=cBLUE if label=="Left" else cGREEN
            filled_rect(frame,4,yp-22,tw+20,yp+10,(10,10,10),0.65)
            cv2.putText(frame,text,(10,yp),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2,cv2.LINE_AA)
    filled_rect(frame,0,h-22,len(_HINT)*6,h,(10,10,10),0.6)
    cv2.putText(frame,_HINT,(6,h-6),cv2.FONT_HERSHEY_PLAIN,0.78,cGREY,1,cv2.LINE_AA)

def save_screenshot(frame):
    os.makedirs("screenshots",exist_ok=True)
    path=f"screenshots/hand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(path,frame); print(f"[Screenshot] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=_MAX_HANDS,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.50,
        )
    )

    cam=CamReader(0); w,h=cam.wh

    try:
        import pyautogui; sw,sh=pyautogui.size()
    except:
        sw,sh=1920,1080

    hand_states: list[HandState]=[HandState() for _ in range(_MAX_HANDS)]
    rgb_buf=np.empty((h,w,3),dtype=np.uint8)

    # Rolling FPS with running sum (O(1))
    fps_buf=deque(maxlen=30)
    fps_sum=0.; fps=0.; prev_t=time.perf_counter()

    vol_ctl    = VolumeController()
    bright_ctl = BrightnessController()
    mouse_ctl  = MouseController(sw,sh)
    canvas     = DrawingCanvas(w,h)
    trainer    = GestureTrainer()
    media_ctl  = MediaController()

    show_fingers=True; show_ids=False; show_gesture=True; show_bbox=False
    show_debug=False; show_terminal=False; show_speed=False; show_angles=False
    show_shake=False; show_2hand=True
    vol_mode=False; bright_mode=False; mouse_mode=False
    canvas_mode=False; media_mode=False
    term_detail=TERM_KEYPTS; term_frame_n=0
    recorder: Optional[Recorder]=None
    two_hand_label=""; two_hand_hold=0
    media_label=""

    print("Ready.\n"+_HINT)

    while True:
        raw=cam.read(); frame=cv2.flip(raw,1)
        cv2.cvtColor(frame,cv2.COLOR_BGR2RGB,dst=rgb_buf)
        ts_ms=int(time.perf_counter()*1000)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_buf)
        result=detector.detect_for_video(mp_img,ts_ms)

        det_labels=[]
        for i in range(len(result.hand_landmarks)):
            raw_label=result.handedness[i][0].display_name
            det_labels.append("Right" if raw_label=="Left" else "Left")

        matched=match_hands(hand_states,result.hand_landmarks,det_labels)

        hands_info=[]; terminal_data=[]; all_arrs=[]; all_states=[]
        now_t=time.perf_counter()

        if canvas_mode: canvas.blend_onto(frame)

        for slot_i,(hs,det_i) in enumerate(matched):
            raw_lm=result.hand_landmarks[det_i]
            label=det_labels[det_i]; is_right=(label=="Right")

            # Per-hand buffer — no shared buffer corruption
            arr=lm_to_array(raw_lm, slot_i)
            arr=hs.smoother.update(arr)
            pts=pts_to_pixels(arr,w,h)

            hs.last_wrist=arr[0,:2].copy(); hs.label=label; hs.active=True
            hs.speed_track.update(arr[0,:2],now_t)
            hs.shake_det.update(arr[0,:2])

            raw_st=finger_states_np(arr,is_right)
            states=hs.debouncer.update(raw_st)
            n_fing=int(states.sum())
            all_arrs.append(arr); all_states.append(states)

            if show_gesture:
                cust=trainer.classify(states,arr)
                if cust:
                    hs.g_emoji="⭐"; hs.g_name=cust; hs.hold=GESTURE_HOLD
                else:
                    e,n=classify_gesture(states,arr,is_right)
                    if n: hs.g_emoji=e; hs.g_name=n; hs.hold=GESTURE_HOLD
                    elif hs.hold>0:
                        hs.hold-=1
                        if hs.hold==0: hs.g_emoji=""; hs.g_name=""

            if recorder: recorder.write_positions(arr,pts,label,hs.g_name,n_fing)

            draw_skeleton(frame,pts,is_right)
            draw_tips(frame,pts,states)
            if show_bbox:    draw_bbox(frame,pts,w,h,is_right,label)
            if show_ids:     draw_ids(frame,pts)
            if show_gesture: draw_gesture(frame,pts,hs.g_emoji,hs.g_name,is_right)
            if show_debug:   draw_debug(frame,label,raw_st,states,slot_i)
            if show_angles:  draw_angle_readout(frame,arr,h,w)
            if show_speed:   draw_speed_overlay(frame,pts,hs.speed_track.speed,
                                                hs.speed_track.accel,is_right,slot_i)
            if show_shake:   draw_shake_indicator(frame,pts,hs.shake_det.shaking,
                                                   hs.shake_det.intensity)
            if vol_mode:
                lvl=vol_ctl.set_from_pinch(float(np.linalg.norm(arr[4,:2]-arr[8,:2])))
                draw_pinch_bar(frame,pts,lvl,"VOL",cGREEN,0)
            if bright_mode:
                lvl=bright_ctl.set_from_pinch(float(np.linalg.norm(arr[4,:2]-arr[8,:2])))
                draw_pinch_bar(frame,pts,lvl,"BRI",cYELLOW,52)
            if mouse_mode:
                _,clicked=mouse_ctl.update(arr)
                draw_mouse_overlay(frame,pts,clicked,is_right)
            if canvas_mode:
                canvas.update(pts,states)
            if media_mode:
                media_label=media_ctl.update(arr,states)

            if not show_bbox:
                col=cGREEN if is_right else cBLUE
                cv2.putText(frame,label,(int(pts[0,0])-28,int(pts[0,1])+40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.75,col,2,cv2.LINE_AA)

            hands_info.append((label,n_fing))
            terminal_data.append((label,arr,pts,states,hs.g_name,n_fing))

        # Multi-hand gesture
        if show_2hand and len(all_arrs)==2:
            tg=classify_two_hand_gesture(all_arrs,all_states)
            if tg: two_hand_label=tg; two_hand_hold=20
            elif two_hand_hold>0:
                two_hand_hold-=1
                if two_hand_hold==0: two_hand_label=""
        else: two_hand_label=""
        if show_2hand: draw_two_hand_gesture(frame,two_hand_label,h,w)
        if media_mode: draw_media_badge(frame,media_label,w,h)
        draw_canvas_hud(frame,canvas,canvas_mode,w)

        # O(1) FPS with running sum
        now=time.perf_counter(); dt=max(now-prev_t,1e-9)
        new_fps=1./dt
        if len(fps_buf)==fps_buf.maxlen: fps_sum-=fps_buf[0]
        fps_buf.append(new_fps); fps_sum+=new_fps
        fps=fps_sum/len(fps_buf); prev_t=now

        # Mode indicator
        modes={"VOL":vol_mode,"BRI":bright_mode,"MOUSE":mouse_mode,
               "CANVAS":canvas_mode,"MEDIA":media_mode,"SPEED":show_speed,
               "ANGLES":show_angles,"SHAKE":show_shake}
        draw_mode_panel(frame,modes,w)
        draw_hud(frame,h,fps,hands_info,show_fingers,recorder is not None)

        if show_terminal:
            print_terminal(term_frame_n,fps,terminal_data,term_detail)
            term_frame_n+=1

        if recorder: recorder.write_frame(frame); recorder.end_frame()

        cv2.imshow("Hand Tracking",frame)
        key=cv2.waitKey(1)&0xFF

        if   key==ord("q"): break
        elif key==ord("s"): save_screenshot(frame)
        elif key==ord("r"):
            if recorder is None: recorder=Recorder(w,h)
            else: recorder.stop(); recorder=None
        elif key==ord("f"): show_fingers=not show_fingers
        elif key==ord("l"): show_ids=not show_ids
        elif key==ord("g"): show_gesture=not show_gesture
        elif key==ord("b"): show_bbox=not show_bbox
        elif key==ord("d"): show_debug=not show_debug
        elif key==ord("v"): show_speed=not show_speed
        elif key==ord("a"): show_angles=not show_angles
        elif key==ord("z"): show_shake=not show_shake
        elif key==ord("h"): show_2hand=not show_2hand
        elif key==ord("n"):
            media_mode=not media_mode
            print(f"[Media] {'ON  point=play/pause  swipe=skip' if media_mode else 'OFF'}")
        elif key==ord("m"):
            mouse_mode=not mouse_mode
            print(f"[Mouse] {'ON' if mouse_mode else 'OFF'}"
                  +("" if mouse_ctl.available else " — install pyautogui"))
        elif key==ord("c"):
            canvas_mode=not canvas_mode
            print(f"[Canvas] {'ON' if canvas_mode else 'OFF'}")
        elif key==ord("x"):
            canvas.save() if canvas_mode else canvas.clear()
        elif key==ord("t"):
            show_terminal=not show_terminal
            if show_terminal: sys.stdout.write("\n"*44); sys.stdout.flush()
        elif key==ord("p"): term_detail=(term_detail+1)%3
        elif key==ord("`"):
            vol_mode=not vol_mode
            print(f"[Volume] {'ON' if vol_mode else 'OFF'}"
                  +("" if vol_ctl.available else " — failed to init, check pycaw install"))
        elif key==ord("\\"):
            bright_mode=not bright_mode
            print(f"[Brightness] {'ON' if bright_mode else 'OFF'}"
                  +("" if bright_ctl.available else " — failed to init"))
        elif key==ord("["):
            print("[Trainer] Hold a gesture, then type a name:")
            name=input("  Name: ").strip()
            if name and terminal_data:
                _,a,_,s,_,_=terminal_data[0]; trainer.record(s,a,name)
        elif key==ord("]"): trainer.delete_last()

    if recorder: recorder.stop()
    detector.close(); cam.release(); cv2.destroyAllWindows()
    if _IS_WIN:
        try:
            import comtypes; comtypes.CoUninitialize()
        except: pass
    print("Stopped.")

if __name__=="__main__":
    main()