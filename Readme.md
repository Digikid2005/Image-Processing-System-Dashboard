# Athena Vision Dashboard

A Flask-based computer vision command center that integrates YOLOv8 for spatial obstruction detection and Real-ESRGAN for asynchronous super-resolution image enhancement. 

## System Setup Instructions

Follow these 10 steps to deploy the application on a fresh Ubuntu Linux environment.

1. Clone the Repository: Pull the project from GitHub and navigate into the root project directory.

2. Install System Dependencies: Run sudo apt update and install essential Linux packages: python3, python3-venv, python3-pip, libgl1, libglib2.0-0, and mysql-server.

3. Setup the Virtual Environment: Create and activate an isolated Python environment using python3 -m venv venv and source venv/bin/activate.

4. Install Python Libraries: Execute pip install -r requirements.txt to install all necessary Python frameworks and AI dependencies.

5. Configure Global Storage: Create the required centralized directories by running sudo mkdir -p /opt/athena_vision/data/videos (repeat for images, enhanced, and analyzed).

6. Assign Directory Permissions: Grant your local user read/write access to the storage folder by running sudo chown -R $USER:$USER /opt/athena_vision and sudo chmod -R 755 /opt/athena_vision.

7. Initialize the Database: Run python init_db.py to establish the SQLAlchemy schema and provision the default administrative account.

8. Download AI Weights: Ensure the pre-trained RealESRGAN_x4plus.pth file is downloaded and present in the root directory.

9. Boot the AI Worker: In your active terminal, run python worker.py to start the background enhancement engine.

10. Launch the Command Center: Open a second terminal window, activate the virtual environment, and run python app.py to start the web server on http://localhost:5000

## How to download ESRGAN model:

Run this in terminal:
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth



### Access the Command Center


URL: http://localhost:5000

Default Username: admin

Default Password: admin123




## NEW CHANGES INTRODUCED:

### models.py :
 
1. It creates a centralized db object that both our Flask application and worker script can bind to.

2. It maps the web_users and captured_images tables to Python classes.

3. It adds a created_at / timestamp column to both tables. This is an industry-standard practice for auditing and sorting data that was missing from the original raw SQL schema.

### NEW DIRECTORY TO ACCESS THE IMAGES

1. The files are now stored in /opt/athena_vision/data, which is outside the standard home directory, the easiest way to manage and view them is to create a Symbolic Link (Shortcut) directly on your Ubuntu Desktop.


## PTZ Camera Integration Strategy

1. Video Streaming (Already Solved): Enterprise PTZ cameras (e.g., Hikvision, Dahua, Axis) broadcast their video using the RTSP (Real-Time Streaming Protocol). Because we routed cv2.VideoCapture through a background thread, your application can already accept RTSP URLs directly (e.g., rtsp://admin:password@192.168.1.100:554/stream1). The system will process it exactly like the Android IP Webcam.

2. PTZ Mechanical Control (Future Implementation): To actually move the camera (Pan, Tilt, Zoom) from your dashboard, you will need to implement the ONVIF protocol. When you are ready to add control buttons to the UI, we will install the onvif-zeep Python library. We will map UI arrow buttons to API endpoints in app.py that send specific vector commands (e.g., MoveUp, ZoomIn) directly to the camera's IP address.

## Network Camera & PTZ Integration Architecture

The Athena Vision Command Center natively supports external network cameras, including mobile IP webcams and enterprise PTZ (Pan-Tilt-Zoom) hardware, utilizing a resilient multi-threaded ingestion pipeline.

### Supported Streaming Protocols
* **HTTP/MJPEG:** Utilized primarily for mobile device IP webcam applications.
* **RTSP (Real-Time Streaming Protocol):** The industry standard for enterprise security cameras (H.264/H.265 encoded).

### Asynchronous Buffer Management
To prevent network latency, packet loss, or dropped frames from stalling the primary Flask application thread, network feeds are ingested via a dedicated `CameraStream` background thread. 
* The thread forces `cv2.CAP_PROP_BUFFERSIZE = 1`.
* It continuously drains the network buffer, ensuring the DSP engine only evaluates the absolute most recent frame.
* If a network dropout occurs, the thread implements a generic back-off (`time.sleep(0.1)`) to prevent CPU thrashing until the connection is restored.

### Future PTZ Control Expansion
While video ingestion is protocol-agnostic, future mechanical control of PTZ hardware will be implemented using the **ONVIF** Profile S standard. Future updates will introduce the `onvif-zeep` library to send SOAP requests to the camera hardware for mechanical manipulation directly from the web dashboard.














