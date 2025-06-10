## Disclaimer
This repository is part of a university assignment and is provided for demonstration purposes only. Do not redistribute or reuse the code without permission.

# Image Generation Tool Optimised for Gaussian Splatting

A Blender-based framework for generating optimised multi-view datasets and camera matrices for 3D Gaussian Splatting reconstruction.

<p align="center">
  <img src="examples/guitar_output.png" width="400"/>
  <img src="examples/sofa_output.png" width="400"/>
</p>

## Project Overview

This repository implements a systematic Blender-based pipeline for acquiring multiple calibrated views of a 3D object under controlled illumination conditions. The framework generates both high-re[...]

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
git clone https://github.com/vl-pixel00/blender-view-gen.git
cd blender-view-gen
```

2. Execute the script via Blender's command-line interface:

#### With GUI (Recommended for Initial Setup)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --python gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --python gaussian_blender_script.py -- ^
  --object_path="C:\path\to\model.obj" ^
  --output_dir="C:\path\to\output"

# Linux
blender --python gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"
```

#### Headless Operation (For Batch Processing or Remote Execution)

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --background --python gaussian_blender_script.py -- \
  --object_path="/path/to/model.obj" \
  --output_dir="/path/to/output"

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --background --python gaussian_blender_script.py -- ^
  --object_path="C:\path\to\model.obj" ^
  --output_dir="C:\path\to\output"

# Linux
blender --background --python gaussian_blender_script.py -- \
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

> **Important**: You must replace the placeholder path in `start_blender.py` with the actual path to your virtual environment's site-packages directory. This is required for Blender to access your external Python libraries.

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
/Applications/Blender.app/Contents/MacOS/Blender --background --python start_blender.py -- --run gaussian_blender_script.py

# Windows
"C:\Program Files\Blender Foundation\Blender\blender.exe" --background --python start_blender.py -- --run gaussian_blender_script.py

# Linux
blender --background --python start_blender.py -- --run gaussian_blender_script.py
```

5. For GUI mode, once the Blender interface initialises with the external libraries loaded:
   - Navigate to the "Scripting" workspace via the top toolbar
   - Open the `gaussian_blender_script.py` file
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
3. Open `gaussian_blender_script.py` from the file browser
4. Locate the `Args` class near the end of the script (around line 771) and modify the default values as needed:

```python
class Args:
    def __init__(self):
        self.object_path = "./models/example_model.glb"  # Change to your actual 3D asset path
        self.output_dir = "./output"                    # Change to your desired output path
        self.num_renders = 36
        self.engine = "BLENDER_EEVEE_NEXT"
        self.resolution = 2048
        self.use_area_lights = True
        self.separate_matrices_dir = True
```

> **Note**: When running the script directly from Blender's Script Editor, you must modify these default paths to point to your actual model and desired output location.

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

## Visual Workflow Overview

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
└─────────────────┘     └────────────────────────└──────────────────┘
```

## Experimental Reproducibility

To ensure consistent results across experimental setups:

- Standardise Blender version (e.g., 4.3.2)
- Maintain uniform view count (`--num_renders 36`)
- Use consistent resolution (`--resolution 2048`)
- Select identical rendering engine (`--engine BLENDER_EEVEE_NEXT`)
- Apply equivalent lighting methodology (`--use_area_lights True`)
- Process outputs with consistent Gaussian Splatting implementation

## ComfyUI Workflow for Image-to-3D Generation

This repository also incorporates a ComfyUI workflow for algorithmically generating 3D assets from textual specifications or reference imagery through integration of SDXL, SAM, and Hunyuan 3D 2.0 Multi-View technologies.

<p align="center">
  <img src="examples/Image_to_3D_ComfyUI_workflow.png" width="800"/>
</p>

### ComfyUI Installation Guide

ComfyUI is a powerful node-based interface for creating stable diffusion workflows. This section provides step-by-step instructions for installing ComfyUI and setting up the required custom nodes for the Image-to-3D generation workflow.

#### Installation Steps

##### Windows Installation

1. Clone the ComfyUI repository:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install PyTorch with CUDA support and other dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. Create a custom nodes directory and install ComfyUI Manager:
   ```bash
   mkdir -p custom_nodes
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   cd ..
   ```

5. Launch ComfyUI:
   ```bash
   python main.py
   ```

##### macOS Installation

1. Clone the ComfyUI repository:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install PyTorch with MPS acceleration and other dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

4. Create a custom nodes directory and install ComfyUI Manager:
   ```bash
   mkdir -p custom_nodes
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   cd ..
   ```

5. Launch ComfyUI:
   ```bash
   python main.py
   ```

##### Linux Installation

1. Clone the ComfyUI repository:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install PyTorch with CUDA support and other dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. Create a custom nodes directory and install ComfyUI Manager:
   ```bash
   mkdir -p custom_nodes
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   cd ..
   ```

5. Launch ComfyUI:
   ```bash
   python main.py
   ```

### Installing Required Custom Nodes

The Image-to-3D workflow requires specific custom node packages that can be installed using the ComfyUI Manager:

1. Access the ComfyUI web interface at http://localhost:8188
2. Click on the "Manager" tab in the interface
3. Search for and install the following custom nodes:
   - **ComfyUI Segment Anything**: Search for "segment anything" and install
   - **ComfyUI-Hunyuan-3D**: Search for "hunyuan" and install
   - **ComfyUI-Advanced-ControlNet**: Search for "advanced controlnet" and install

Alternatively, you can manually install these repositories in the `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/storyicon/comfyui_segment_anything
git clone https://github.com/your-username/ComfyUI-Hunyuan-3D
git clone https://github.com/your-username/ComfyUI-Advanced-ControlNet
```
## Exemplification of Three-Dimensional Reconstruction Methodology

<p align="center">
  <table>
    <tr>
      <td width="50%" style="vertical-align: top;">
        <img src="examples/table_058.png" width="100%" alt="Input Image of Table"/>
        <br>
        <img src="examples/table_058_parameters.png" width="100%" alt="Parameters Used for Table Reconstruction"/>
      </td>
      <td width="50%">
        <img src="examples/table_058_3Dvideo.gif" width="100%" alt="3D Model Animation"/>
      </td>
    </tr>
    <tr>
      <td align="center"><b>Input Image with Parameters</b></td>
      <td align="center"><b>Generated 3D Model (Animated)</b></td>
    </tr>
  </table>
</p>

### Utilising the Image-to-3D Workflow

1. Launch ComfyUI by running `python main.py` from the ComfyUI directory
2. Access the web interface at http://localhost:8188
3. Import the workflow:
   - Click the "Load" button in the top menu
   - Select the workflow JSON file (to be provided separately)
4. Configure the workflow parameters:
   - Adjust the text prompt to specify your desired subject
   - Modify segmentation parameters if needed
   - Set the number of views to generate (8-24 recommended)
   - Specify the output directory for the generated assets
5. Execute the workflow by clicking the "Queue Prompt" button
6. The workflow will generate:
   - High-fidelity base images from text prompts
   - Segmented subject images with transparent backgrounds
   - Multi-view perspectives suitable for 3D reconstruction

### Workflow Architecture

The workflow implements a sequential pipeline integrating three computational models:

1. **SDXL Module**:
   - Generates high-fidelity images from text prompts
   - Implements style control via textual conditioning
   - Outputs resolution-optimised base images for segmentation

2. **SAM (Segment Anything) Module**:
   - Performs automatic segmentation on generated imagery
   - Implements background removal with alpha channel preservation
   - Outputs isolated subject assets for 3D reconstruction

3. **Hunyuan 3D 2.0 MV Module**:
   - Processes segmented imagery into multi-view perspectives
   - Generates consistent camera viewpoints with appropriate transformations
   - Outputs multi-angle perspectives suitable for 3D reconstruction algorithms

### Technical Considerations

- **Memory Optimisation**: If you encounter VRAM issues, try reducing the resolution parameters in the workflow
- **Initial Execution Latency**: First-time execution may involve downloading model files, which can take time
- **Segmentation Refinement**: For challenging subjects, you may need to adjust the segmentation parameters
- **Output Integration**: The multi-view images generated by this workflow can be directly used with the Blender-based pipeline described earlier in this README

## Repository Structure

- `examples/` – Sample visual output generated by the Gaussian Splatting pipeline
- `start_blender.py` – Launches Blender with external Python environment integration
- `gaussian_blender_script.py` – Main script for rendering multi-view images and exporting camera matrices
- `requirements.txt` – Python dependency list for setting up the virtual environment
- `README.md` – Project setup, usage instructions, and technical documentation

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Blender Python API Documentation](https://docs.blender.org/api/current/index.html)
- [Objaverse-XL Rendering Source](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering)
- [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
- [Segment Anything Model Technical Paper](https://arxiv.org/abs/2304.02643)
- [ComfyUI Segment Anything Integration](https://github.com/storyicon/comfyui_segment_anything)
- [Stable Diffusion XL Technical Report](https://arxiv.org/abs/2307.01952)
- [Hunyuan 3D: Generalised 3D Generation Framework](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [ComfyUI Manager Implementation](https://github.com/ltdrdata/ComfyUI-Manager)