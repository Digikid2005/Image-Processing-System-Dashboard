from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import mysql.connector
import cv2
from ultralytics import YOLO
import os
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import threading
import glob

app = Flask(__name__)
app.secret_key = "athena_super_secret_key"

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'videos')
IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
ENHANCED_DIR = os.path.join(BASE_DIR, 'static', 'enhanced')
ANALYZED_DIR = os.path.join(BASE_DIR, 'static', 'analyzed')

for folder in [UPLOAD_FOLDER, IMAGES_DIR, ENHANCED_DIR, ANALYZED_DIR]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': 'password123',
    'database': 'athena_vision'
}

# --- GLOBAL VARIABLES ---
system_logs = ["[SYSTEM] Athena Vision Command Center initialized."]
camera = None
current_frame = None
last_alert_time = 0
video_source = 0 
soiling_accumulator = None 
bg_subtractor = None 

print("Loading YOLOv8 Object Detector...")
yolo_model = YOLO('yolov8n.pt')

# --- UTILITY FUNCTIONS ---
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    system_logs.append(f"[{timestamp}] {message}")
    if len(system_logs) > 50:
        system_logs.pop(0)

# --- FINAL UNIFIED ML/DSP CAMERA HEALTH ENGINE (3-STATE) ---
def check_camera_health_dsp(frame, accumulator, bg_subtractor):
    try:
        # 1. PRE-PROCESSING & ROI
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_h, roi_w = gray.shape

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv_roi[:, :, 1])
        mean_intensity = np.mean(gray)
        std_dev = np.std(gray) 

        # Night detection
        is_true_night = avg_saturation < 15.0 and mean_intensity < 140
        env_mode = "NIGHT_IR" if is_true_night else "DAY"

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_gray = clahe.apply(gray)
        denoised_gray = cv2.bilateralFilter(norm_gray, d=9, sigmaColor=75, sigmaSpace=75)

        # ==========================================
        # STATE 1: VIEW OBSTRUCTED
        # ==========================================
        
        # A. Total Washout / Pitch Black
        if std_dev < 15.0:
            if mean_intensity < 50:
                return "obstructed", 99.0, 100.0, accumulator, bg_subtractor
            elif mean_intensity > 200:
                return "rain/smudge/dust/blur", 95.0, 100.0, accumulator, bg_subtractor
        # B. AI Proximity Obstruction (Masks, Hands, Vehicles blocking lens)
        try:
            small_frame = cv2.resize(frame, (320, 320))
            # conf=0.15 catches faint detections on heavy letterboxing
            results = yolo_model.predict(small_frame, verbose=False, classes=[0, 2, 5, 7], conf=0.15)
            frame_area = 320 * 320

            max_coverage = 0.0
            max_conf = 0.0
            is_obstructed = False

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy()) # Identify the object type

                obj_area = (x2 - x1) * (y2 - y1)
                coverage_pct = (obj_area / frame_area) * 100

                # DYNAMIC THRESHOLDS:
                # Class 0 (Person): 12.0% limit (Handles letterboxing and camera tampering)
                # Classes 2,5,7 (Vehicles): 45.0% limit (Allows normal foreground parking)
                threshold = 12.0 if cls_id == 0 else 45.0

                if coverage_pct > threshold and coverage_pct > max_coverage:
                    max_coverage = coverage_pct
                    max_conf = float(box.conf[0].cpu().numpy()) * 100
                    is_obstructed = True

            if is_obstructed:
                if max_conf < 50.0: max_conf = min(99.0, max_conf + 40.0)
                if max_conf > 50.0:
                    return "obstructed", max_conf, max_coverage, accumulator, bg_subtractor
        except Exception as e: pass

        # ==========================================
        # STATE 2: RAIN / SMUDGE / DUST / BLUR
        # ==========================================

        # A. Severe Weather / Dust Storm (Dark Channel Prior)
        if env_mode == "DAY":
            try:
                dcp_frame = cv2.resize(frame, (160, 120))
                dark_pass = np.amin(dcp_frame, axis=2)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                dark_channel = cv2.erode(dark_pass, kernel)
                dcp_mean = np.mean(dark_channel)
                
                if dcp_mean > 110: 
                    conf = min(99.0, (dcp_mean / 255.0) * 100)
                    coverage_pct = min(100.0, dcp_mean - 10)
                    return "rain/smudge/dust/blur", conf, coverage_pct, accumulator, bg_subtractor
            except Exception as e: pass

        # B. Grid-Based Blur & Haze
        grid_rows, grid_cols = 4, 4
        cell_h = roi_h // grid_rows
        cell_w = roi_w // grid_cols
        
        sharp_cells, blurry_cells, valid_cells = 0, 0, 0
        base_threshold = 25.0 if env_mode == "NIGHT_IR" else 60.0
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                cell = denoised_gray[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                if np.mean(cell) < 25: continue
                valid_cells += 1
                cell_lap_var = cv2.Laplacian(cell, cv2.CV_64F).var()
                
                if cell_lap_var > base_threshold: sharp_cells += 1
                elif cell_lap_var < (base_threshold * 0.4): blurry_cells += 1
                    
        if valid_cells > 0:
            sharp_ratio = sharp_cells / valid_cells
            blurry_ratio = blurry_cells / valid_cells
            
            if sharp_ratio <= 0.25 and blurry_ratio >= 0.20:
                coverage_pct = blurry_ratio * 100
                if env_mode == "DAY" and std_dev > 45.0 and blurry_ratio < 0.4:
                    pass
                else:
                    return "rain/smudge/dust/blur", 90.0, coverage_pct, accumulator, bg_subtractor

        # C. Spatial Rain Drop & Smudge (Local Energy Masking)
        rain_gray = cv2.GaussianBlur(norm_gray, (3, 3), 0)
        lap = cv2.Laplacian(norm_gray, cv2.CV_64F)
        lap_abs = cv2.convertScaleAbs(lap)
        local_energy = cv2.boxFilter(lap_abs, cv2.CV_8U, (15, 15))
        
        energy_thresh = 40 if env_mode == "NIGHT_IR" else 75
        _, complex_bg_mask = cv2.threshold(local_energy, energy_thresh, 255, cv2.THRESH_BINARY)
        # Scalpel dilation to preserve dense sheets of rain
        complex_bg_mask = cv2.dilate(complex_bg_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        if env_mode == "NIGHT_IR":
            edges = cv2.Canny(rain_gray, 80, 200) 
            trigger_limit = 2.5
        else:
            v = np.median(rain_gray)
            lower = int(max(15, (1.0 - 0.5) * v)) 
            upper = int(min(255, (1.0 + 0.5) * v))
            edges = cv2.Canny(rain_gray, lower, upper)
            trigger_limit = 1.8 
            
        clean_edges = cv2.bitwise_and(edges, cv2.bitwise_not(complex_bg_mask))
        
        kernel_length = 15
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        diag1_kernel = np.eye(kernel_length, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        
        h_lines = cv2.morphologyEx(clean_edges, cv2.MORPH_OPEN, horizontal_kernel)
        v_lines = cv2.morphologyEx(clean_edges, cv2.MORPH_OPEN, vertical_kernel)
        d1_lines = cv2.morphologyEx(clean_edges, cv2.MORPH_OPEN, diag1_kernel)
        d2_lines = cv2.morphologyEx(clean_edges, cv2.MORPH_OPEN, diag2_kernel)
        
        structural_lines = cv2.bitwise_or(h_lines, v_lines)
        structural_lines = cv2.bitwise_or(structural_lines, d1_lines)
        structural_lines = cv2.bitwise_or(structural_lines, d2_lines)
        
        droplet_edges = cv2.bitwise_and(clean_edges, cv2.bitwise_not(structural_lines))
        
        contours, _ = cv2.findContours(droplet_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_smudge_pixels = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 15000:
                valid_smudge_pixels += area
        
        smudge_pct = (valid_smudge_pixels / (roi_h * roi_w)) * 100
        confidence = min(smudge_pct * 10, 99.0)

        if smudge_pct > trigger_limit and confidence > 25.0:
            return "rain/smudge/dust/blur", confidence, smudge_pct, accumulator, bg_subtractor
            
        # ==========================================
        # STATE 3: CAMERA CLEAR
        # ==========================================
        return "clean", 100.0, 0.0, accumulator, bg_subtractor

    except Exception as e:
        print(f"DSP Error: {e}")
        return "unknown", 0.0, 0.0, accumulator, bg_subtractor

# --- VIDEO STREAMING GENERATOR ---
def generate_frames():
    global current_frame, last_alert_time, video_source, soiling_accumulator, bg_subtractor
    
    local_source = None
    camera = None
    is_image = False
    static_img = None
    
    while True:
        if local_source != video_source:
            local_source = video_source
            if camera: camera.release()
            
            if isinstance(local_source, str) and local_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                is_image = True
                static_img = cv2.imread(local_source)
                add_log("Streaming static image as video feed...")
            else:
                is_image = False
                camera = cv2.VideoCapture(local_source)
        
        if is_image:
            if static_img is None: 
                time.sleep(0.5)
                continue
            frame = static_img.copy()
            time.sleep(0.1) 
        else:
            if camera is None or not camera.isOpened(): 
                time.sleep(0.5)
                continue
            success, frame = camera.read()
            if not success:
                if isinstance(local_source, str):
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue
        
        current_frame = frame.copy()
        current_time = time.time()


        # Health Monitor (Runs every 2 seconds)
        if current_time - last_alert_time > 2.0:

            # Unpack all 5 variables returned by the Unified DSP engine
            status, conf, coverage, soiling_accumulator, bg_subtractor = check_camera_health_dsp(frame, soiling_accumulator, bg_subtractor)

            # --- DYNAMIC COLOR CODING ---
            if coverage >= 75.0:
                color = "#ff4444" # Red
            elif coverage >= 50.0:
                color = "#ff8800" # Orange
            elif coverage >= 15.0:
                color = "#ffcc00" # Yellow
            else:
                color = "#00ff00" # Green

            # Dashboard Logging with inline HTML colors
            if status == "obstructed":
                add_log(f'<span style="color: {color};">DSP CRITICAL: Camera View Obstructed! ({conf:.1f}% conf | {coverage:.1f}% coverage)</span>')
            elif status == "rain/smudge/dust/blur":
                add_log(f'<span style="color: {color};">DSP ALERT: Rain/Smudge/Dust/Blur Detected ({conf:.1f}% conf | {coverage:.1f}% coverage)</span>')
            elif status == "clean":
                add_log(f'<span style="color: {color};">DSP Status: Camera Clear</span>')

            last_alert_time = current_time



        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- WEB ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM web_users WHERE username = %s AND password_hash = %s", (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['logged_in'] = True
            session['username'] = user['username']
            add_log(f"User '{username}' logged in.")
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid Credentials."
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    return jsonify({"logs": system_logs})

@app.route('/api/latest_results')
def latest_results():
    if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
    
    response_data = {"original": None, "enhanced": None, "analyzed": None}
    
    enhanced_files = glob.glob(os.path.join(ENHANCED_DIR, '*_EDSR.jpg'))
    if enhanced_files:
        latest_sr = max(enhanced_files, key=os.path.getctime)
        sr_filename = os.path.basename(latest_sr)
        orig_filename = sr_filename.replace("_EDSR.jpg", ".jpg")
        
        response_data["enhanced"] = f"/static/enhanced/{sr_filename}"
        if os.path.exists(os.path.join(IMAGES_DIR, orig_filename)):
            response_data["original"] = f"/static/images/{orig_filename}"

    analyzed_files = glob.glob(os.path.join(ANALYZED_DIR, 'analyzed_*.jpg'))
    if analyzed_files:
        latest_analyzed = max(analyzed_files, key=os.path.getctime)
        response_data["analyzed"] = f"/static/analyzed/{os.path.basename(latest_analyzed)}"

    return jsonify(response_data)

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_source, soiling_accumulator, bg_subtractor
    if 'media_file' not in request.files: return redirect(url_for('dashboard'))
    file = request.files['media_file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        add_log(f"Media uploaded: {filename}. Switching feed...")
        video_source = filepath
        soiling_accumulator = None 
        bg_subtractor = None
    return redirect(url_for('dashboard'))

@app.route('/api/live_camera', methods=['POST'])
def live_camera():
    global video_source, soiling_accumulator, bg_subtractor
    add_log("Switching to USB Live Webcam feed...")
    video_source = 0
    soiling_accumulator = None 
    bg_subtractor = None
    return jsonify({"status": "success"})

# --- AI ACTION PIPELINES ---
@app.route('/api/screenshot', methods=['POST'])
def capture_screenshot():
    global current_frame
    if current_frame is None: return jsonify({"error": "No video feed"}), 400
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img_{timestamp}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    cv2.imwrite(filepath, current_frame)
    add_log(f"Screenshot saved: {filename}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO captured_images (filename, file_path, is_processed, is_analyzed) VALUES (%s, %s, FALSE, FALSE)", (filename, filepath))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "image": filename})

@app.route('/api/super_resolution', methods=['POST'])
def trigger_sr():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM captured_images WHERE is_processed = FALSE LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row:
        add_log(f"Sent Image ID {row['id']} to dedicated AI Worker for EDSR enhancement...")
    else:
        add_log("ℹ️ No new screenshots in queue for the AI Worker.")
    return jsonify({"status": "queued"})

@app.route('/api/detect_objects', methods=['POST'])
def trigger_detection():
    def run_det():
        add_log("Starting YOLOv8 Object Detection...")
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM captured_images WHERE is_analyzed = FALSE LIMIT 1")
            row = cursor.fetchone()
            
            if row:
                add_log(f"Analyzing Image ID {row['id']}...")
                img = cv2.imread(row['file_path'])
                results = yolo_model(img, verbose=False)
                annotated_img = results[0].plot()
                
                new_filename = "analyzed_" + row['filename']
                new_filepath = os.path.join(ANALYZED_DIR, new_filename)
                cv2.imwrite(new_filepath, annotated_img)
                
                cursor.execute("UPDATE captured_images SET is_analyzed = TRUE WHERE id = %s", (row['id'],))
                conn.commit()
                add_log(f"Analyzed image saved as {new_filename}")
            else:
                add_log("ℹ️ No unanalyzed images found.")
            conn.close()
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc().splitlines()[-1]
            add_log(f"WARNING !!! YOLO Thread Crash: {error_msg}")
    threading.Thread(target=run_det).start()
    return jsonify({"status": "processing"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
