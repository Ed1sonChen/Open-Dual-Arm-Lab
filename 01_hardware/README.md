# 🔧 Hardware

This is where the dual-arm setup begins.

Before there is calibration, control, perception, or learning, there is hardware: two robot arms, grippers, a workstation, and a network that makes the whole system talk. This section documents the physical backbone of our dual-arm platform and how we assembled it into a working research setup.

The goal is simple: build a reliable system that can grow into a platform for **bimanual manipulation**.

---

<p align="center">
  <img src="Assets/images/hardware.png" alt="dual-arm setup overview" width="60%">
</p>

## ⚙️ What powers this setup?

Our current platform is built around four core pieces:

- 🤖 **Two xArm 6 robot arms**
- ✋ **Grippers for object interaction**
- 💻 **A dedicated workstation computer**
- 🌐 **A wired network centered around a switch**

Together, these components form the first working version of the lab's dual-arm infrastructure.

---

## 🤖 Robot Arms

At the center of the system are **two xArm 6** robot arms.

We chose xArm 6 because it offers a practical balance between reach, flexibility, compactness, and ease of deployment in a tabletop research setup. For a dual-arm platform, that matters: the arms need to be capable enough for coordinated manipulation, but also compact enough to share a workspace safely.

These two arms are the foundation for future tasks such as:

- 🤝 bimanual coordination
- 🧷 grasp-and-hold behaviors

In other words, these are not just two independent manipulators placed side by side. They are being assembled as a **shared platform for interaction**.

---

## ✋ Grippers

Each arm is equipped with a gripper for physical interaction with objects.

The grippers are what turn the setup from a motion platform into a manipulation platform. They support early experiments in grasping and stabilization, and later enable more complex behaviors such as object handoff, regrasping, and coordinated dual-arm interaction.

**Current gripper configuration**
- Left arm: **xArm Gripper G2**
- Right arm: **xArm Gripper G2**

This section will be expanded later with more details on mounting, interfaces, and grasping capabilities.

---

## 💻 Workstation Computer

The entire setup is controlled from a dedicated workstation.

This machine serves as the main control and development hub for the system. It is used for robot communication, experiment scripting, logging, and future integration of perception and learning modules.

**Current workstation configuration**
- 🧠 CPU: **AMD Ryzen 9 9950X**
- 🎮 GPU: **NVIDIA RTX 4090**
- 💾 Storage: **Samsung 4TB 990 PRO PCIe 4.0 NVMe M.2 SSD**
- 🐧 Operating System: **Ubuntu 22.04**

Using a dedicated workstation makes the setup easier to maintain, easier to debug, and easier to scale as more components are added.

---

## 🌐 How everything is connected

To make the dual-arm setup work reliably in the lab, we organized the system around a **network switch**.

The connection is built as follows:

1. 🧱 The **switch** connects to the **wall Ethernet port** in the lab.
2. 🤖 Both **xArm 6 robot arms** connect directly to the **switch**.
3. 💻 The **workstation computer** connects to the **switch** through an Ethernet cable.
4. 📡 A **router** is also connected to the **switch**.

This gives the setup a central wired communication structure that is easy to manage and easy to expand.

### 🗺️ Network topology

```text
🏫 Wall Ethernet Port
          │
        🔀 Switch
    ┌─────┼─────┬─────┐
    🤖     🤖     💻     📡
  Arm 1  Arm 2    PC   Router