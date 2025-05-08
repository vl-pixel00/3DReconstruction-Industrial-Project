# Gaussian Splat Image Generation Tool

A Blender-based framework for generating optimised multi-view datasets and camera matrices for 3D Gaussian Splatting reconstruction.

![Gaussian Splat Example](https://repository-images.githubusercontent.com/667382847/b8c90618-15d6-4231-bd22-f5c8584de616)

## Project Overview

This repository implements a systematic Blender-based pipeline for acquiring multiple calibrated views of a 3D object under controlled illumination conditions. The framework generates both high-resolution images and corresponding camera intrinsic/extrinsic matrices in `.npy` format, specifically formatted for Gaussian Splatting algorithmic applications.

The procedural workflow encompasses:

1. Importation and positioning of the target `.obj` model within Blender's coordinate system  
2. Strategic camera distribution via Fibonacci sphere algorithm for optimal spatial sampling  
3. Implementation of consistent illumination parameters via configurable light sources  
4. High-fidelity image rendering using either EEVEE-Next or Cycles rendering engines  
5. Export of both rendered perspectives and corresponding camera transformation matrices  
6. Structured organisation of outputs within a hierarchical directory with JSON metadata

## Technical Requirements

- Blender 4.0+ (validated with Blender 4.3.2)
- Python 3.11+ (included with Blender installation)
- NumPy (required for matrix operations)
- Torch and Pillow (optional dependencies)

## Installation & Configuration

### Method 1: Utilising Blender's Internal Python Interpreter

This approach leverages Blender's built-in Python environment without external dependencies.

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gaussian-splat-generator.git
cd gaussian-splat-generator
```

2. Execute the script via Blender's command-line interface:

#### With GUI (Recommended for Initial Setup)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --python improved_gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --python improved_gaussian_blender_script.py -- ^
  --object_path="C:\path\to\model.obj" ^
  --output_dir="C:\path\to\output"

# Linux
blender --python improved_gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"
```

#### Headless Operation (For Batch Processing or Remote Execution)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --background --python improved_gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --background --python improved_gaussian_blender_script.py -- ^
  --object_path="C:\path\to\model.obj" ^
  --output_dir="C:\path\to\output"

# Linux
blender --background --python improved_gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"
```

### Method 2: Integrating External Python Virtual Environment (Recommended)

This methodological approach facilitates Blender's access to external Python libraries while maintaining GUI functionality, essential for complex numerical operations and visualisation.

The repository includes a `start_blender.py` script that enables integration of external Python libraries into Blender's execution environment.

1. Create and activate a dedicated virtual environment:

```bash
python3.11 -m venv blender-env
source blender-env/bin/activate           # macOS/Linux
blender-env\Scripts\activate              # Windows
```

2. Install the required dependencies:

```bash
pip install numpy==1.26.4 pillow==10.2.0 torch==2.2.0
```

3. Modify the `start_blender.py` script with your environment-specific site-packages path:

```python
import sys

# Replace with the absolute path to your virtual environment's site-packages directory
sys.path.append('/Users/yourusername/.pyenv/versions/blender-env/lib/python3.11/site-packages')

# Verification of successful library integration
import numpy as np
print("NumPy version inside Blender:", np.__version__)

import bpy
print("Blender is running this script successfully.")
```

To determine your site-packages path, execute:
```bash
python -c "import site; print(site.getsitepackages()[0])"
```

4. Launch Blender with the environment integration script:

#### With GUI (Recommended for Development and Verification)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --python start_blender.py

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --python start_blender.py

# Linux
blender --python start_blender.py
```

#### Headless Operation (For Computational Clusters or Automated Workflows)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --background --python start_blender.py -- --run improved_gaussian_blender_script.py

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --background --python start_blender.py -- --run improved_gaussian_blender_script.py

# Linux
blender --background --python start_blender.py -- --run improved_gaussian_blender_script.py
```

5. For GUI mode, once the Blender interface initialises with the external libraries loaded:
   - Navigate to the "Scripting" workspace via the top toolbar
   - Open the `improved_gaussian_blender_script.py` file
   - Configure the script parameters as detailed in the "Interactive Usage" section
   - Execute the script via the "Run Script" button or Alt+P key combination

## Execution Parameters

### Command-Line Arguments

```
--object_path               Path to the target 3D model file (.obj) [required]
--output_dir                Directory for rendered images and matrices [required]
--num_renders               Number of viewpoints to generate [default: 36]
--engine                    Rendering engine: "CYCLES" or "BLENDER_EEVEE_NEXT" [default: BLENDER_EEVEE_NEXT]
--resolution                Output image resolution in pixels [default: 2048]
--use_area_lights           Enable area lights for enhanced illumination [default: True]
--separate_matrices_dir     Store camera matrices in dedicated directory [default: True]
```

### Interactive Blender Usage

1. Ensure Blender has launched with the `start_blender.py` script (Method 2)
2. Navigate to the "Scripting" workspace in the Blender interface
3. Open `improved_gaussian_blender_script.py` from the file browser
4. Locate or implement the `Args` class and modify the parameters accordingly:

```python
class Args:
    def __init__(self):
        self.object_path = "/path/to/model.obj"  # Specify model path
        self.output_dir = "/path/to/output"      # Define output directory
        self.num_renders = 36
        self.engine = "BLENDER_EEVEE_NEXT"
        self.resolution = 2048
        self.use_area_lights = True
        self.separate_matrices_dir = True
```

5. Execute the script via the interface button or Alt+P keyboard shortcut
6. Monitor execution progress via the console output:
   - Windows: Window → Toggle System Console
   - macOS/Linux: Terminal window from which Blender was launched

## Output Data Structure

```
output_dir/
├── 000.png, 001.png, ..., 035.png        # Multi-view rendered images
├── matrices/                             # Camera transformation matrices
│   ├── 000.npy, 001.npy, ..., 035.npy
├── cameras.json                          # Consolidated camera parameters
└── README.txt                            # Output specification
```

## Technical Implementation Features

- **Optimised View Distribution**: Fibonacci sphere algorithm with polar filtering for uniform spatial sampling
- **Enhanced Illumination**: Calibrated lighting system with increased intensity (4x for sun lights, 2x for area lights)
- **Multiple Rendering Engines**: Support for Cycles and EEVEE-Next with appropriate parameter optimisation
- **Automatic GPU Utilisation**: Dynamic detection of available graphics hardware (Metal, CUDA, or CPU fallback)
- **Standardised Matrix Export**: Camera matrices in NumPy format with consistent coordinate transformations
- **Compositor Deactivation**: Prevention of extraneous directory generation (e.g., png.47 folders)
- **Interface Flexibility**: Support for both interactive and headless execution workflows

## Troubleshooting Guidelines

### Environment Integration Issues

#### Library Import Failures

1. Validate the virtual environment path in `start_blender.py`:
   ```python
   # Obtain the correct path for your system
   python -c "import site; print(site.getsitepackages()[0])"
   
   # Update the script with the precise path
   sys.path.append('/absolute/path/to/site-packages')
   ```

2. Confirm installation of required packages within the virtual environment:
   ```bash
   # Activate the environment
   source blender-env/bin/activate  # macOS/Linux
   blender-env\Scripts\activate     # Windows
   
   # Verify package installation
   pip list | grep numpy
   pip list | grep torch
   ```

3. Ensure Python version compatibility:
   ```bash
   # Check virtual environment Python version
   python --version
   
   # Verify Blender's Python version
   blender --python-expr "import sys; print(sys.version)"
   ```

### Blender-Specific Issues

#### AttributeError in Render Settings

Different Blender versions implement varied attribute structures. The script includes version compatibility checks, but additional adaptations may be necessary:

```python
if hasattr(scene.eevee, "gi_diffuse_bounces"):
    scene.eevee.gi_diffuse_bounces = 3
```

#### Unwanted Directory Generation

Ensure compositor deactivation prior to rendering:

```python
bpy.context.scene.use_nodes = False
```

#### GPU Rendering Failures

- **macOS**: Verify Metal configuration (Blender Preferences → System → GPU Backend)
- **Windows/Linux**: Ensure GPU driver compatibility with CUDA or OptiX
- The script implements automatic CPU fallback if GPU acceleration is unavailable

## Process Workflow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Virtual         │     │ Blender          │     │ Script           │
│ Environment     │────>│ Launch with      │────>│ Execution with   │
│ Configuration   │     │ Library Injection│     │ Parameters       │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                               │                          │
                               │                          ▼
┌─────────────────┐     ┌──────┘                 ┌──────────────────┐
│ Results         │<────┤                        │ Automated        │
│ Analysis        │     │                        │ Rendering        │
└─────────────────┘     └────────────────────────┘
```

## Experimental Reproducibility

To ensure consistent results across experimental setups:

- Standardise Blender version (e.g., 4.3.2)
- Maintain uniform view count (`--num_renders 36`)
- Use consistent resolution (`--resolution 2048`)
- Select identical rendering engine (`--engine BLENDER_EEVEE_NEXT`)
- Apply equivalent lighting methodology (`--use_area_lights True`)
- Process outputs with consistent Gaussian Splatting implementation

## Repository Structure

- `start_blender.py` - Environment integration script for external library access
- `gaussian_blender_script.py` - Primary multi-view generation implementation
- `requirements.txt` - Dependency specification for virtual environment
- `README.md` - Technical documentation

## Dependencies

```
numpy==1.26.4
pillow==10.2.0
torch==2.2.0
```

## License

MIT License

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Blender Python API Documentation](https://docs.blender.org/api/current/index.html)
