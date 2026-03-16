# 📦 Data Collection

This section documents the data collection pipeline for our dual-arm robotics setup.

Our current data collection system is designed for **teleoperated demonstration recording**. It combines:

- **Meta Quest 3** based VR teleoperation
- **xArm robot state logging**
- **front and wrist RealSense RGB recording**
- **teleoperation command recording**
- **episode-level metadata saving**
- **camera calibration snapshot saving**

---

## ✨ Overview

The current data collection pipeline records a full teleoperated episode while controlling the robot in real time.

During an episode, the system logs:

- robot joint states
- robot TCP pose
- gripper position
- teleoperation commands from VR
- front camera RGB frames
- wrist camera RGB frames
- episode metadata
- camera calibration files

---

## 🧠 Current Setup

At this stage, the current data collection script supports:

- **Meta Quest 3 teleoperation**
- **one xArm robot**
- **two Intel RealSense RGB cameras**
  - front camera
  - wrist camera
- **keyboard-triggered episode control**

Although this repository follows a broader dual-arm roadmap, the current collection script is focused on a single teleoperated manipulation stream. 

---

## 🗂️ What the Script Does

The data collection script performs the following functions:

### 1. Robot Teleoperation
The robot is teleoperated through the Quest 3 controller stream received through WebSockets.

### 2. Robot State Recording
At a fixed frequency, the script records:

- 6 joint angles
- 6 joint velocities
- TCP position
- TCP orientation
- gripper position

### 3. Teleoperation Command Recording
The script logs commanded robot motion and raw VR controller signals, including:

- commanded TCP pose
- commanded gripper state
- VR position
- VR quaternion
- trigger value
- grip value

### 4. Camera Recording
Two RealSense RGB cameras are recorded:

- **front camera**
- **wrist camera**

Frames are saved as image files with timestamps.

### 5. Calibration Snapshot
At the start of each episode, camera calibration JSON files are copied into the episode folder.

### 6. Episode Metadata
Each episode stores metadata such as:

- episode ID
- task name
- object ID
- success / failure
- failure reason
- operator
- start and end time

---

## 🧰 Main Components in the Code

The current script includes the following major components:

- **Configuration block**  
  Stores robot IP, camera serials, limits, control frequencies, camera settings, gripper range, and home pose.

- **One Euro Filter 3D**  
  Used to smooth teleoperation target positions and rotations.

- **Helper functions**  
  Utilities for timestamps, clamping motion deltas, saving JSON, creating directories, image overlays, and visualization.

- **EpisodeRecorder**  
  Handles episode folder creation, CSV writing, metadata saving, calibration copying, and image saving.

- **RealSenseReader**  
  Starts each RealSense camera, applies manual settings, saves calibration, and continuously reads the latest frame.

- **Robot state recording thread**  
  Logs joint states, joint velocities, TCP pose, and gripper position.

- **Camera recording thread**  
  Saves front and wrist RGB images into the active episode.

- **Camera display thread**  
  Shows front and wrist RGB streams side by side with frame ID and timestamps.

- **High-frequency robot control loop**  
  Applies filtered teleoperation targets to the xArm.

- **VR packet processing**  
  Converts Quest controller motion into robot translation, rotation, and gripper commands.

- **Keyboard control loop**  
  Lets the operator home the robot, start recording, stop with success, stop with failure, or quit.

- **WebSocket server**  
  Receives Quest 3 teleoperation data and drives the robot in real time.

---

## 📂 Output Folder Structure

The dataset is currently saved under:
```bash
dataset/raw/
```

Each recorded episode is stored in its own folder:
```bash
dataset/raw/
├── camera_calibration/
│   ├── front_<serial>.json
│   └── wrist_<serial>.json
├── episode_000001/
│   ├── front_rgb/
│   │   ├── 000000_<timestamp>.png
│   │   └── ...
│   ├── wrist_rgb/
│   │   ├── 000000_<timestamp>.png
│   │   └── ...
│   ├── camera_calibration/
│   │   ├── front_<serial>.json
│   │   └── wrist_<serial>.json
│   ├── robot_state.csv
│   ├── teleop_command.csv
│   └── episode_meta.json
├── episode_000002/
└── ...
```

---

## ⌨️ Keyboard Controls

The script supports simple **keyboard-based episode control**.

### Available commands

- `h` → return robot to home pose  
- `r` → start recording  
- `s` → stop recording and mark episode as success  
- `f` → stop recording and mark episode as failure  
- `q` → quit the program  

---