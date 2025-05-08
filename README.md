# Masterclass Img-to-3D 

## Blender Setup

This repository contains a Python script developed for academic coursework. It is designed to run inside Blender's **Scripting workspace**, using Blender’s internal Python environment (Python 3.11.9). The script uses `numpy`, `pillow`, and `torch`, and supports hardware acceleration via **Apple’s Metal (MPS)** or **NVIDIA CUDA** on compatible systems.

---

## Project Details
 
- **Environment:** Blender 4.0+, Python 3.11.9  
- **Platform:** macOS (primary), compatible with Windows and Linux  
- **Execution Mode:** GUI-only (not headless)  
- **Dependencies:** Installed into Blender's internal Python  
- **Hardware Acceleration:** Supports MPS (macOS) and CUDA (Linux/Windows)

---

## Quick Setup Guide

Follow these steps to replicate the development environment and run the script inside Blender.

---

### 1. Locate Blender’s Internal Python

#### macOS:

/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11

#### Windows:

C:\Program Files\Blender Foundation\Blender 4.0\4.0\python\bin\python.exe

#### Linux:

/usr/share/blender/4.0/python/bin/python3.11

---

### 2. Enable pip

Run the following in a terminal:

```bash
"<path-to-blender-python>" -m ensurepip
 "<path-to-blender-python>" -m pip install --upgrade pip setuptools wheel
Replace <path-to-blender-python> with the appropriate path from Step 1.

---

Make sure:
- The `bash` block has **exactly three backticks** to open and close
- You don't nest explanatory text *inside* the code block

This fix will prevent GitHub from treating the rest of the document as part of the command block.

Would you like me to lint and finalise your entire `README.md` one more time to guarantee formatting safety across all platforms?
