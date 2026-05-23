# Athena Vision Dashboard

A Flask-based computer vision command center that integrates YOLOv8 for spatial obstruction detection and Real-ESRGAN for asynchronous super-resolution image enhancement. 

## System Setup Instructions

Follow these 10 steps to deploy the application on a fresh Ubuntu Linux environment.

### Step 1: Clone the Repository
Download the project files to your local machine and navigate into the working directory.
```bash
git clone <your_github_repository_url>
cd Athena_ProjectX/code_repo



### STEP-2:
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv libgl1 libglib2.0-0 mysql-server -y


### Step 3: Configure the MySQL Database:
sudo mysql
CREATE DATABASE IF NOT EXISTS athena_vision;
USE athena_vision;

CREATE TABLE IF NOT EXISTS web_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50),
    password_hash VARCHAR(255)
);

INSERT IGNORE INTO web_users (id, username, password_hash) VALUES (1, 'admin', 'admin123');

CREATE TABLE IF NOT EXISTS captured_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    file_path VARCHAR(255),
    is_processed BOOLEAN DEFAULT FALSE,
    is_analyzed BOOLEAN DEFAULT FALSE
);

ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'password123';
FLUSH PRIVILEGES;
EXIT;


### Step-4: Create and Activate Virtual Enviroment:
python3 -m venv venv
source venv/bin/activate

Execute the Deployment Script:

chmod +x setup_env.sh
./setup_env.sh

### Step-5: Download AI Model Weights:

wget [https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)


### Step-6: Verify Directory Structure:

mkdir -p static/videos static/images static/enhanced static/analyzed

### Step-7: Launch the AI Background Worker
The system requires a dedicated background process to handle heavy GPU/CPU tensor computations. Start the worker in your active terminal:
python worker.py


Launch the Web Dashboard:
In another terminal:

source venv/bin/activate
python app.py

### Step-8: Access the Command Center


URL: http://localhost:5000

Default Username: admin

Default Password: admin123



















