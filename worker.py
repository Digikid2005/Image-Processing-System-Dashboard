import time
import os
import cv2
import sys
import logging
from flask import Flask
from models import db, CapturedImage

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [WORKER] %(message)s', datefmt='%H:%M:%S')

# Intercept outdated import request and redirect to modern torchvision module
import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- GLOBAL CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENHANCED_DIR = '/opt/athena_vision/data/enhanced'
os.makedirs(ENHANCED_DIR, exist_ok=True)

# Create a headless Flask context to bind SQLAlchemy outside of the web server
worker_app = Flask(__name__)
worker_app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:password123@localhost/athena_vision'
worker_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(worker_app)

# --- LOAD AI MODELS ---
logging.info("Initializing AI Worker and loading Real-ESRGAN Engine...")

model_path = os.path.join(BASE_DIR, 'RealESRGAN_x4plus.pth')
if not os.path.exists(model_path):
    logging.error(f"Weights not found at {model_path}! Please ensure the model file is present.")
    sys.exit(1)

logging.info("Loading pre-trained Real-ESRGAN weights...")
rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=rrdb_model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)
logging.info(f"Worker Ready! Running on: {device}. Listening for tasks...")

def process_queue():
    """Queries the database for unprocessed images and runs the enhancement pipeline."""
    with worker_app.app_context():
        # Fetch the oldest unprocessed image using SQLAlchemy ORM
        task = CapturedImage.query.filter_by(is_processed=False).order_by(CapturedImage.id.asc()).first()
        
        if task:
            logging.info(f"Task found! Enhancing Image ID {task.id}...")
            filepath = task.file_path
            
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                
                # Hardware optimization: resize massive inputs before upscale
                height, width, _ = img.shape
                max_width = 480 
                if width > max_width:
                    scale = max_width / width
                    new_w, new_h = int(width * scale), int(height * scale)
                    img = cv2.resize(img, (new_w, new_h))
                    logging.info(f"Resized input to {new_w}x{new_h} to optimize memory.")

                # Real-ESRGAN Execution
                logging.info("Running Real-ESRGAN 4x Upscaling...")
                result_bgr, _ = upsampler.enhance(img, outscale=4)
                
                # Post-Processing: Denoise & Contrast
                logging.info("Applying Denoising and Contrast Enhancement...")
                result_bgr = cv2.fastNlMeansDenoisingColored(result_bgr, None, 3, 3, 7, 21)

                lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                limg = cv2.merge((cl, a, b))
                result_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
                # Save processed file to global directory
                new_filename = task.filename.replace(".jpg", "_EDSR.jpg")
                new_filepath = os.path.join(ENHANCED_DIR, new_filename)
                cv2.imwrite(new_filepath, result_bgr)
                
                # Update database record using ORM
                task.is_processed = True
                db.session.commit()
                logging.info(f"Success! Saved enhanced image as {new_filename}")
            else:
                logging.warning(f"File missing on disk: {filepath}. Skipping and marking as processed.")
                task.is_processed = True
                db.session.commit()

if __name__ == '__main__':
    while True:
        try:
            process_queue()
            time.sleep(2)
        except Exception as e:
            logging.error(f"Worker Exception: {e}")
            time.sleep(5)