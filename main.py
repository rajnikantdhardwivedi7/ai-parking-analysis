"""
Parking Management System — Aerial/Top-Down Edition
Uses Background Subtraction + Contour Detection (works perfectly for aerial footage)
+ YOLO as bonus layer
Usage: python main.py cars.mp4
"""

import subprocess, sys, os, shutil, time

# ── Auto-install ───────────────────────────────────────────────────────────────
for pkg in ["ultralytics", "opencv-python", "numpy"]:
    try:
        __import__(pkg.replace("-","_").split(".")[0])
    except ImportError:
        print(f"[setup] Installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import cv2, numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# AERIAL CAR DETECTOR — pure OpenCV, no YOLO needed
# Works by: grayscale → adaptive threshold → contour detection
# Cars from above = bright/dark rectangular blobs on uniform grey tarmac
# ══════════════════════════════════════════════════════════════════════════════
class AerialCarDetector:
    def __init__(self, frame_w, frame_h):
        self.W = frame_w
        self.H = frame_h
        # Expected car size at 1280x720 aerial view
        # Cars look ~60-120px tall, 40-80px wide from directly above
        self.min_area = 800
        self.max_area = 25000
        self.min_ar   = 0.3    # min aspect ratio (w/h)
        self.max_ar   = 3.5    # max aspect ratio
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=True)
        self.frame_count = 0
        self.ref_gray = None   # used for static difference method

    def detect(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        boxes = []

        # ── METHOD 1: Background subtraction (catches moving cars) ────────────
        fg = self.bg.apply(frame)
        fg[fg == 127] = 0   # remove shadows
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
        fg = cv2.dilate(fg, k, iterations=3)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area < a < self.max_area:
                x,y,w,h = cv2.boundingRect(c)
                ar = w/max(h,1)
                if self.min_ar < ar < self.max_ar:
                    boxes.append((x,y,x+w,y+h,"bg"))

        # ── METHOD 2: Color/texture blob detection (catches PARKED cars) ─────
        # Parked cars have different color from grey tarmac
        # Convert to HSV and look for non-grey regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:,:,1]   # saturation
        v_ch = hsv[:,:,2]   # brightness

        # Cars are either colorful (high sat) or very bright/dark vs tarmac
        # Tarmac = low saturation, mid-brightness grey
        car_mask = np.zeros(gray.shape, dtype=np.uint8)

        # High saturation = colored car
        _, sat_mask = cv2.threshold(s_ch, 35, 255, cv2.THRESH_BINARY)

        # Very bright OR very dark vs tarmac
        mean_v = int(np.median(v_ch))
        bright_mask = cv2.threshold(v_ch, min(mean_v+50, 220), 255, cv2.THRESH_BINARY)[1]
        dark_mask   = cv2.threshold(v_ch, max(mean_v-60, 20),  255, cv2.THRESH_BINARY_INV)[1]

        car_mask = cv2.bitwise_or(sat_mask, bright_mask)
        car_mask = cv2.bitwise_or(car_mask, dark_mask)

        # Remove road markings (thin white lines) with erosion
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
        car_mask = cv2.erode(car_mask, k_erode, iterations=1)
        k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, k_close)

        cnts2,_ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts2:
            a = cv2.contourArea(c)
            if self.min_area < a < self.max_area:
                x,y,w,h = cv2.boundingRect(c)
                ar = w/max(h,1)
                if self.min_ar < ar < self.max_ar:
                    # filter out top road strip and very thin lane lines
                    if y < self.H * 0.05: continue
                    boxes.append((x,y,x+w,y+h,"cv"))

        # ── NMS across all methods ────────────────────────────────────────────
        if not boxes:
            return []
        return self._nms(boxes, 0.3)

    def _nms(self, boxes, thresh):
        if not boxes: return []
        arr  = np.array([[b[0],b[1],b[2],b[3]] for b in boxes], dtype=float)
        tags = [b[4] for b in boxes]
        x1,y1,x2,y2 = arr[:,0],arr[:,1],arr[:,2],arr[:,3]
        areas = (x2-x1)*(y2-y1)
        order = areas.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w   = np.maximum(0, xx2-xx1)
            h   = np.maximum(0, yy2-yy1)
            inter = w*h
            iou_v = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order = order[np.where(iou_v<=thresh)[0]+1]
        return [(int(arr[i,0]),int(arr[i,1]),int(arr[i,2]),int(arr[i,3]),tags[i]) for i in keep]


# ══════════════════════════════════════════════════════════════════════════════
# YOLO — bonus layer for extra detections (conf very low)
# ══════════════════════════════════════════════════════════════════════════════
def load_yolo():
    try:
        from ultralytics import YOLO
        print("[info] Loading YOLOv8s...")
        m = YOLO("yolov8s.pt")
        print("[info] YOLO ready ✓")
        return m
    except Exception as e:
        print(f"[warn] YOLO failed: {e}")
        return None

def yolo_detect(model, frame, W, H):
    if model is None: return []
    try:
        res = model(frame, verbose=False, conf=0.08, iou=0.30,
                    imgsz=1280, agnostic_nms=True, classes=[2,3,5,7])[0]
        boxes = []
        for b in res.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            if y1 < H*0.05: continue
            boxes.append((x1,y1,x2,y2,"yolo"))
        return boxes
    except:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PARKING SLOTS — matching your video layout exactly
# Your video: cars park in VERTICAL columns, driving lanes between columns
# ══════════════════════════════════════════════════════════════════════════════
def make_slots(W, H):
    """
    From your video screenshot:
    - ~8 columns of parking (some empty, some full)
    - Each column has ~9-10 spots (rows)
    - Driving lanes separate the columns
    - Top ~8% = road, skip it
    """
    slots = []

    top     = int(H * 0.08)   # skip road at top
    bot     = int(H * 0.98)
    left    = int(W * 0.01)
    right   = int(W * 0.99)

    usable_w = right - left
    usable_h = bot - top

    # Each car from above: ~85px wide, ~130px tall at 1280x720
    slot_w = 82
    slot_h = 128
    col_gap = max(4, (usable_w - 8*slot_w) // 9)   # driving lane width approx
    row_gap = 3

    # Figure out how many columns and rows fit
    n_cols = max(1, usable_w // (slot_w + col_gap))
    n_rows = max(1, usable_h // (slot_h + row_gap))

    total_grid_w = n_cols * slot_w + (n_cols-1) * col_gap
    start_x = left + (usable_w - total_grid_w) // 2

    for c in range(n_cols):
        for r in range(n_rows):
            x1 = start_x + c*(slot_w+col_gap)
            y1 = top      + r*(slot_h+row_gap)
            x2 = x1 + slot_w
            y2 = y1 + slot_h
            slots.append((x1,y1,x2,y2))

    return slots


# ══════════════════════════════════════════════════════════════════════════════
# OCCUPANCY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def is_occupied(slot, all_boxes):
    sx1,sy1,sx2,sy2 = slot
    sw = sx2-sx1; sh = sy2-sy1; s_area = sw*sh
    for b in all_boxes:
        bx1,by1,bx2,by2 = b[0],b[1],b[2],b[3]
        # centre-point inside slot (best for aerial)
        cx=(bx1+bx2)//2; cy=(by1+by2)//2
        if sx1<=cx<=sx2 and sy1<=cy<=sy2:
            return True
        # IoU check
        ix1=max(sx1,bx1); iy1=max(sy1,by1)
        ix2=min(sx2,bx2); iy2=min(sy2,by2)
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
        inter=iw*ih
        if inter==0: continue
        b_area=(bx2-bx1)*(by2-by1)
        union=s_area+b_area-inter
        if inter/max(union,1) >= 0.12:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_slot(frame, slot, occupied):
    x1,y1,x2,y2 = slot
    color = (0,200,0) if occupied else (0,0,200)   # green=occupied, red=empty
    ov = frame.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,0.22,frame,0.78,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)

def draw_car(frame, box):
    x1,y1,x2,y2,src = box
    col = (0,255,255) if src=="yolo" else (255,200,0)
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
    lbl = "car"
    fs=0.38
    (tw,lh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,fs,1)
    cv2.rectangle(frame,(x1,max(0,y1-lh-5)),(x1+tw+4,y1),(255,255,255),-1)
    cv2.putText(frame,lbl,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),1,cv2.LINE_AA)

def draw_ui(frame, n_occ, n_free, n_cars, fps, W):
    # Title bar
    cv2.rectangle(frame,(0,0),(W,30),(15,15,15),-1)
    cv2.putText(frame,"Parking Management System",(W//2-145,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),1,cv2.LINE_AA)
    # Info panel top-right
    panel = [
        (f"Parked cars: {n_occ}",  (0,220,0)),
        (f"Available  : {n_free}", (0,0,220)),
        (f"Detected   : {n_cars}", (0,200,255)),
        (f"FPS        : {fps:.1f}",(160,160,160)),
    ]
    pw=215; rh=32; px=W-pw-10; py=34
    for i,(txt,col) in enumerate(panel):
        by1=py+i*rh; by2=by1+rh-2
        cv2.rectangle(frame,(px,by1),(px+pw,by2),(25,25,25),-1)
        cv2.rectangle(frame,(px,by1),(px+pw,by2),(70,70,70),1)
        cv2.rectangle(frame,(px,by1),(px+5,by2),col,-1)
        cv2.putText(frame,txt,(px+12,by1+rh-9),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESS
# ══════════════════════════════════════════════════════════════════════════════
def compress(src, dst):
    ff=shutil.which("ffmpeg")
    if ff:
        tmp=src+".tmp.mp4"
        try:
            r=subprocess.run([ff,"-y","-i",src,"-vcodec","libx264","-crf","23",
                              "-preset","fast","-movflags","+faststart","-an",tmp],
                             capture_output=True)
            if r.returncode==0 and os.path.exists(tmp):
                os.replace(tmp,dst)
                if src!=dst and os.path.exists(src): os.remove(src)
                return True
        except: pass
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
    if src!=dst: os.replace(src,dst)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv)<2:
        print("Usage: python main.py cars.mp4"); sys.exit(1)
    inp = sys.argv[1]
    if not os.path.exists(inp):
        print(f"[error] Not found: {inp}"); sys.exit(1)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[error] Cannot open: {inp}"); sys.exit(1)

    VW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_V = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output at 1280x720 for speed
    OW, OH = 1280, 720
    scale_x = OW/VW; scale_y = OH/VH

    print(f"[info] Source: {VW}x{VH} | Output: {OW}x{OH} | {TOTAL} frames")

    # Load detectors
    cv_det  = AerialCarDetector(OW, OH)
    yolo    = load_yolo()

    # Build parking slots
    slots = make_slots(OW, OH)
    print(f"[info] Parking slots: {len(slots)}")

    RAW = "output_raw.mp4"
    wri = cv2.VideoWriter(RAW, cv2.VideoWriter_fourcc(*"mp4v"), FPS_V, (OW,OH))

    t0 = time.time()
    print(f"[info] Processing {TOTAL} frames...\n")

    for fi in range(TOTAL):
        ret, frame = cap.read()
        if not ret: break

        ft = time.time()

        # Resize to output size
        frame = cv2.resize(frame,(OW,OH))

        # ── Detect cars ────────────────────────────────────────────────────────
        cv_boxes   = cv_det.detect(frame)
        yolo_boxes = yolo_detect(yolo, frame, OW, OH)

        # merge — NMS across both sources
        all_raw = cv_boxes + yolo_boxes
        if all_raw:
            arr    = np.array([[b[0],b[1],b[2],b[3]] for b in all_raw],dtype=float)
            scores = np.ones(len(all_raw))
            blist  = [[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in all_raw]
            idxs   = cv2.dnn.NMSBoxes(blist,[float(s) for s in scores],0.01,0.35)
            if len(idxs):
                all_boxes = [all_raw[i] for i in idxs.flatten()]
            else:
                all_boxes = []
        else:
            all_boxes = []

        # ── Occupancy ──────────────────────────────────────────────────────────
        occ_flags = [is_occupied(s, all_boxes) for s in slots]
        n_occ     = sum(occ_flags)
        n_free    = len(slots)-n_occ

        # ── Draw ───────────────────────────────────────────────────────────────
        for slot,occ in zip(slots,occ_flags):
            draw_slot(frame,slot,occ)

        for b in all_boxes:
            draw_car(frame,b)

        fps_cur = 1.0/max(time.time()-ft,1e-6)
        draw_ui(frame,n_occ,n_free,len(all_boxes),fps_cur,OW)

        wri.write(frame)

        if (fi+1)%30==0 or fi==0:
            avg=(fi+1)/max(time.time()-t0,1e-6)
            pct=(fi+1)/max(TOTAL,1)*100
            print(f"  [{pct:5.1f}%] frame {fi+1}/{TOTAL}  avg={avg:.1f}fps"
                  f"  detected={len(all_boxes)}  occ={n_occ}/{len(slots)}")

    cap.release(); wri.release()
    print(f"\n[info] Done in {time.time()-t0:.1f}s — compressing...")
    ok=compress(RAW,"output.mp4")
    mb=os.path.getsize("output.mp4")/1e6
    print(f"[info] {'H.264' if ok else 'mp4v'} → {mb:.1f} MB")
    print("\nDone. Parking analysis saved to output.mp4")

if __name__=="__main__":
    main()