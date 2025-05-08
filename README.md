# masterclass_img-to-3D_assignment

# Blender Python Script — Setup Guide

This repository contains a Python script designed to run inside **Blender’s Scripting tab**, using Blender’s internal Python environment. It requires three external Python libraries: `numpy`, `pillow`, and `torch`.

The instructions below will help you set everything up — even if you're new to Python or Blender scripting.

---

## Requirements

- Blender 4.0 or newer
- Internet connection (for installing packages)
- macOS, Windows, or Linux
- A terminal or command prompt

---

## Step 1 — Locate Blender’s Internal Python

Blender includes its own isolated version of Python. This is the interpreter you must use to install packages.

### Windows (default):

C:\Program Files\Blender Foundation\Blender 4.0\4.0\python\bin\python.exe

### macOS (default):

/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11

### Linux (example):

---

## Step 2 — Enable pip in Blender’s Python

Blender’s Python usually doesn’t have `pip` enabled by default. This step only needs to be done once.

```bash
"<path-to-blender-python>" -m ensurepip
"<path-to-blender-python>" -m pip install --upgrade pip setuptools wheel

Replace <path-to-blender-python> with the correct path from Step 1.
