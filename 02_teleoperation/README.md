# 🎮 Teleoperation

This section documents the full teleoperation pipeline for our dual-arm setup.

Our current teleoperation interface is built with **Meta Quest 3** and a lightweight **WebXR-based browser interface**. The goal is to enable intuitive VR control for robot teleoperation, debugging, and future demonstration data collection.

At this stage, the teleoperation code is organized under the `WEBXR` folder, which contains:

- `index.html`
- `server.py`

---

## ✨ Overview

The teleoperation workflow is:

1. Prepare the **Meta Quest 3**
2. Connect the Quest and the host workstation properly
3. Create a Python environment on the host machine
4. Install the required dependencies
5. Start a local web server
6. Start the Python backend
7. Open the webpage in the Quest browser
8. Click **ENTER VR**
9. Start teleoperating the robot

---

## 🥽 Hardware

Our current teleoperation hardware is:

- **Meta Quest 3**
- **Host workstation**
- **USB data cable** connecting the Quest 3 to the workstation
- **xArm robot system**

The Quest 3 is used as the VR interface, while the host workstation runs the WebXR frontend and the Python backend that communicates with the robot.

---

## 🌐 Network Requirements

To make the teleoperation system work properly:

- the **Meta Quest 3** and the **host workstation** must be on the **same local network**
- the Quest 3 should also be connected to the workstation with a **USB data cable**
- the workstation must be reachable through its local IP address

> Note: the VR headset and the host computer do **not** use the same IP address.  
> They only need to be on the same local network so the headset browser can access the WebXR webpage hosted by the workstation.

---

## 📱 Meta Quest Setup

Before running the teleoperation interface, make sure the Quest 3 is ready for development use.

### Requirements
- Install the **Meta Horizon** app on your phone
- Enable **Developer Mode** for your Meta Quest device in the Meta Horizon app

### Steps
1. Open the **Meta Horizon** app on your phone
2. Select your connected Quest device
3. Go to **Device Settings**
4. Find **Developer Mode**
5. Turn **Developer Mode** on

---

## 🐍 Python Environment Setup & Teleoperation

We recommend creating an isolated Conda environment to run the teleoperation backend smoothly and avoid dependency conflicts.

### Step 1 — Create and Activate a Conda Environment

Open your terminal and run the following commands to create the environment with Python 3.10 and install all required dependencies:

```bash
# Create the environment
conda create -n teleop python=3.10 -y

# Activate the environment
conda activate teleop

# Install dependencies
pip install websockets numpy scipy xArm-Python-SDK
```

Step 2: Start the teleoperation system
```bash
cd 02_teleoperation/WEBXR
```

Terminal 1 — Start the Web Server
```bash
python -m http.server 8000
```

Terminal 2 — Start the Teleoperation Backend
```bash
python server.py
```

Step3: Launch Teleoperation from Meta Quest 3
Now put on the Meta Quest 3 headset.
1. Open the Quest Browser
Inside the headset, open the Browser application.
2. Enter the WebXR Page
In the address bar, enter:
```bash
http://<HOST_IP>:8000/
```

3. Start VR Mode
Once the page loads:
Scroll to the bottom of the webpage
Click ENTER VR

🎉 You are now ready to teleoperate your robots!