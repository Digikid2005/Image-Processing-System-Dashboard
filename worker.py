import time
import os
import cv2
import mysql.connector

# --- NEW: BASICSR BUG FIX FOR MODERN PYTORCH ---
# This intercepts the outdated import request and redirects it to the modern torchvision module
import sys
import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF
# -----------------------------------------------

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENHANCED_DIR = os.path.join(BASE_DIR, 'static', 'enhanced')
os.makedirs(ENHANCED_DIR, exist_ok=True)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': 'password123',
    'database': 'athena_vision'
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# --- LOAD PRE-TRAINED EDSR MODEL ---
print("⚙️ AI Worker Booting Up...")

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- LOAD PRE-TRAINED REAL-ESRGAN MODEL ---
print("⚙️ AI Worker Booting Up (Real-ESRGAN Engine)...")

model_path = os.path.join(BASE_DIR, 'RealESRGAN_x4plus.pth')
if not os.path.exists(model_path):
    print(f"CRITICAL ERROR: Weights not found at {model_path}!")
    exit()

print("Loading Pre-Trained Real-ESRGAN weights...")
# Define the exact neural network architecture expected by the x4plus weights
rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# Automatically detect if you have a GPU pass-through in WSL, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the inference pipeline
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=rrdb_model,
    tile=400,          # IMPORTANT: Chops the image into 400x400 chunks to protect your RAM
    tile_pad=10,
    pre_pad=0,
    half=False,        # Keep False for CPU stability
    device=device
)
print(f"Worker Ready! Running on: {device}. Listening for tasks...")
def process_queue():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Look for the oldest unprocessed image
    cursor.execute("SELECT * FROM captured_images WHERE is_processed = FALSE ORDER BY id ASC LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        print(f"[{time.strftime('%H:%M:%S')}] Task found! Enhancing Image ID {row['id']}...")
        filepath = row['file_path']
        
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            
            # --- HARDWARE OPTIMIZATION ---
            height, width, _ = img.shape
            max_width = 480 
            if width > max_width:
                scale = max_width / width
                new_w, new_h = int(width * scale), int(height * scale)
                img = cv2.resize(img, (new_w, new_h))
                print(f"Resized input to {new_w}x{new_h} to protect system memory...")

            # --- THE MAGIC AI FIX ---
            print("Running Real-ESRGAN 4x Upscaling (This is heavy, please wait)...")
            # RealESRGAN returns a tuple of (enhanced_image, alpha_channel). We only need the image [0]
            result_bgr, _ = upsampler.enhance(img, outscale=4)
            

            # --- POST-PROCESSING: DENOISING & CONTRAST ---
            print("Applying Denoising and Contrast Enhancement...")
            
            # 1. Denoise (Strength is set to 3 to remove grain but preserve textures)
            result_bgr = cv2.fastNlMeansDenoisingColored(result_bgr, None, 3, 3, 7, 21)

            # 2. Contrast Enhancement (CLAHE)
            # Convert to LAB color space to only enhance the Lightness (L) channel. This prevents color distortion.
            lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # clipLimit controls the contrast intensity (higher = more intense, 2.0 is a good natural baseline)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge back together and convert back to BGR for saving
            limg = cv2.merge((cl, a, b))
            result_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # ---------------------------------------------
            
            # Save and update DB
            new_filename = row['filename'].replace(".jpg", "_EDSR.jpg")
            new_filepath = os.path.join(ENHANCED_DIR, new_filename)
            cv2.imwrite(new_filepath, result_bgr)
            
            cursor.execute("UPDATE captured_images SET is_processed = TRUE WHERE id = %s", (row['id'],))
            conn.commit()
            print(f"Success! Saved perfectly as {new_filename}")
        else:
            print(f"WARNING!!! File missing: {filepath}")
            cursor.execute("UPDATE captured_images SET is_processed = TRUE WHERE id = %s", (row['id'],))
            conn.commit()
            
    conn.close()

if __name__ == '__main__':
    while True:
        try:
            process_queue()
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Worker Error: {e}")
            time.sleep(5)
