# 🤖 Open Dual-Arm Lab

> From two robot arms and a pile of cables to a living dual-arm research platform.

<p align="center">
  <img src="Assets/images/overview.png" alt="dual-arm setup overview" width="60%">
</p>

<p align="center">
  <strong>Building a dual-arm robotics setup from scratch</strong><br>
  hardware · teleoperation · data collection · and the road toward bimanual intelligence
</p>

---

## ✨ What is this repository?

This repository is the public build record of a dual-arm robotics system coming to life.

It starts with the unglamorous but essential pieces:
robot arms, grippers, networking, teleoperation, and data pipelines.

It grows toward something much bigger:
a research platform for **bimanual manipulation**, **interactive data collection**, and future work in **perception, control, and learning**.

This is not a polished end-state repository.

It is the system **while it is being built**.

---

## 🚀 Why build a dual-arm platform from scratch?

Because many robotic demos begin where the real engineering difficulty ends.

A final video may show a successful grasp, a coordinated motion, or a clean manipulation sequence.  
What it usually hides is the much harder question:

**How do you actually build the platform that makes those experiments possible?**

A dual-arm system is not just “two single arms side by side.”  
It requires:

- shared physical workspace
- reliable communication
- teleoperation interfaces
- synchronized data collection
- room for future perception and learning modules
- a system design that can evolve without collapsing under its own complexity

This repository exists to document that process openly.

---

## 🧠 What are we building toward?

At the moment, this project focuses on the engineering foundations of a dual-arm setup.

Over time, it is intended to support research in:

- **teleoperated demonstrations**
- **robot data collection**
- **bimanual manipulation**
- **learning-based robot policies**

In other words, today this repo is about infrastructure.  
Soon it becomes a platform for experiments.  
Eventually, it becomes a platform for ideas.

---

## 🗺️ Repository Map

- [🔧 01_hardware](01_hardware/README.md)
- [🎮 02_teleoperation](02_teleoperation/README.md)
- [📦 03_data_collection](03_data_collection/README.md)
- [⚙️ 04_single_task_models](04_single_task_models/README.md)
- [🛠️ 05_generalist_models](05_generalist_models/README.md)

---

## 📌 Current Status

This project is under active construction.

The system is being developed in layers:

- physical hardware
- control and teleoperation
- data collection infrastructure
- future learning and experimentation

Some modules are already working.  
Some are partially integrated.  
Some are still being designed.

That is intentional.

This repository is meant to capture the build **as a process**, not just as a finished artifact.

---

## 📸 What you will find here

As the platform grows, this repository will gradually include:

- setup photos
- system diagrams
- hardware notes
- teleoperation workflows
- data collection procedures
- milestone demos
- technical decisions and build logs

The goal is to make the platform legible:  
not only what was built, but also **why it was built this way**.

---

## 🔗 Start Here

- [Hardware](01_hardware/README.md)

---

## ⭐ A note on philosophy

Robotics systems do not appear fully formed.

They are assembled, tested, reconfigured, debugged, and rebuilt.  
Every cable routing decision, every network choice, every operator interface, and every data format shapes what the system can eventually become.

This repository is an attempt to treat those engineering decisions as first-class research infrastructure.

Because before there is policy learning, there is setup.  
Before there is autonomy, there is instrumentation.  
Before there is intelligence, there is a system that works.

---

## 🌱 Follow the build

If you are interested in how a real dual-arm robotics platform is designed from the ground up — not only at the level of ideas, but at the level of hardware, interfaces, and infrastructure — this repository is for you.