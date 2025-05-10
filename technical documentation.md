# Gaussian Blender Script: Technical Documentation

## Overview

The `gaussian_blender_script.py` is an 807-line specialised renderer designed specifically for Gaussian Splat generation. It provides deterministic camera placement, calibrated lighting, consistent image output, and robust GPU fallback mechanisms to ensure high-quality data for 3D reconstruction workflows.

This script has been optimised from a more general-purpose 3D rendering tool, with improvements focused on creating photogrammetrically stable, high-resolution images with precise camera metadata required for Gaussian Splatting reconstruction.

## Key Features

- **Deterministic Camera Placement**: Uses a Fibonacci sphere algorithm to ensure even distribution of camera views
- **Enhanced Lighting Systems**: Two configurable five-point lighting setups (sun-based or area-based)
- **High-Resolution Output**: Default 2048×2048 rendering resolution (4× improvement over original)
- **Cross-Platform GPU Acceleration**: Automatic detection and configuration (Metal → CUDA → CPU)
- **Robust Attribute Safety**: Comprehensive checks for Blender version compatibility
- **Simplified Metadata Export**: Compact JSON and NumPy matrix exports for reconstruction software
- **Resilient Rendering Pipeline**: Fallback mechanisms for different rendering engines and hardware

## Implementation Details

### File Import System
```python
IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}
```

The script supports a wide range of 3D file formats through a dictionary-based dispatch system that maps file extensions to the appropriate Blender import operators. Additional format-specific parameters are applied where needed (e.g., `merge_vertices=True` for GLTF/GLB files).

### Scene Normalisation

Before rendering, all models are normalised to fit within a unit cube centred at the world origin. This ensures consistent scale and framing across different models:

```python
def normalize_scene() -> None:
    # Calculate bounding box
    bbox_min, bbox_max = scene_bbox()
    # Scale to fit unit cube
    scale = 1 / max(bbox_max - bbox_min)
    # Apply scale and centre
    for obj in get_scene_root_objects():
        if obj.type != "CAMERA":
            obj.scale = obj.scale * scale
            # ...
```

### Camera Placement

The script places cameras using a Fibonacci sphere algorithm, which provides mathematically even sampling around the object:

```python
def position_camera_for_splat(index: int, total: int, radius: float = 2.0):
    # Using enhanced Fibonacci sphere distribution
    golden_ratio = (1 + 5**0.5) / 2
    i = index
    
    # Distribution formula
    theta = 2 * math.pi * i / golden_ratio
    z = 1 - (2 * (i + 0.5) / total)
    phi = math.acos(z)
    
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
```

This approach prevents clustering at the poles and ensures full spherical coverage of the subject.

### Lighting Systems

The script offers two comprehensive lighting setups:

#### 1. Sun-Based Lighting
Five directional lights (key, fill, rim, bottom, ambient) with enhanced brightness values ranging from 5.0 to 20.0, plus ambient world lighting.

#### 2. Area-Based Lighting
Five area lights strategically positioned to ensure volumetric coverage, with sizes ranging from 3.0 to 8.0 Blender units and enhanced intensities for better surface illumination.

Both systems are calibrated to provide optimal lighting for photogrammetry/Gaussian Splat reconstruction.

### GPU Fallback System

The script includes a robust GPU detection and fallback mechanism:

```python
if sys.platform == "darwin":  # macOS
    device_type = "METAL"
else:
    device_type = "CUDA"  # Default for other platforms
    
try:
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.get_devices()
    try:
        cycles_preferences.compute_device_type = device_type
        log(f"Set GPU device type to {device_type}")
    except Exception as e:
        log(f"Warning: {device_type} not available, using CPU instead: {str(e)}")
        cycles.device = "CPU"
except Exception as e:
    log(f"Warning: Could not configure GPU rendering: {str(e)}, using CPU")
    cycles.device = "CPU"
```

This ensures the script works across different platforms whilst maximising performance.

### Render Configuration

The script configures the rendering engine with safety checks for attribute existence across different Blender versions:

```python
if engine == "BLENDER_EEVEE_NEXT" or engine == "BLENDER_EEVEE":
    if hasattr(bpy.context.scene, 'eevee'):
        eevee = bpy.context.scene.eevee
        # Set higher quality settings - safely check for attributes first
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 64
        if hasattr(eevee, 'use_gtao'):
            eevee.use_gtao = True
            if hasattr(eevee, 'gtao_distance'):
                eevee.gtao_distance = 0.2
        # ...
```

### Matrix Export

Camera matrices are exported in a format compatible with computer vision and 3D reconstruction software:

```python
def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    "Returns the 3x4 RT matrix from the given camera."
    # Use matrix_world to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world
    T_world2bcam = -1 * R_world2bcam @ location

    # Put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT
```

## Performance Improvements

| **Metric**          | **Before** | **After** | **Change** |
|---------------------|------------|-----------|------------|
| Lines of Code       | ~899       | 807       | −10%       |
| Functions           | 25         | 13        | −48%       |
| Classes             | 1          | 0         | −100%      |
| Output Resolution   | 512px      | 2048px    | ×4         |
| Cycles Samples      | 32         | 64        | ×2         |
| GPU Handling        | Manual CUDA| Auto Metal/CUDA/CPU | ↑ portability |
| Metadata Format     | YAML / Verbose JSON | Lean JSON + .npy | ↓ disk usage |
| Compositor          | Enabled    | Disabled   | ↑ stability |

## Usage

### Command Line Usage

```bash
blender --background --python gaussian_blender_script.py -- \
    --object_path "/path/to/3d/model.glb" \
    --output_dir "/path/to/output" \
    --num_renders 36 \
    --resolution 2048 \
    --engine BLENDER_EEVEE_NEXT \
    --use_area_lights \
    --separate_matrices_dir
```

### Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|----------------|
| `--object_path` | string | (Required) | Path to the 3D object file |
| `--output_dir` | string | (Required) | Directory where renders will be saved |
| `--num_renders` | integer | 36 | Number of images to render |
| `--resolution` | integer | 2048 | Width/height of rendered images |
| `--engine` | string | BLENDER_EEVEE_NEXT | Render engine (CYCLES, BLENDER_EEVEE_NEXT) |
| `--use_area_lights` | boolean | True | Use area lights instead of sun lights |
| `--separate_matrices_dir` | boolean | True | Store matrices in separate directory |

### Output Structure

The script generates the following output structure:

```
output_dir/
├── 000.png
├── 001.png
├── ...
├── cameras.json  # Camera metadata
├── README.txt    # Summary info
└── matrices/     # (if separate_matrices_dir=True)
    ├── 000.npy
    ├── 001.npy
    └── ...
```

## Compatibility

- **Blender Versions**: 3.0+ (with appropriate fallbacks for feature differences)
- **Operating Systems**: Windows, macOS, Linux
- **GPU Support**: CUDA, OptiX, Metal, with CPU fallback

## Conclusion

The `gaussian_blender_script.py` script provides a deterministic, platform-aware rendering pipeline optimised for Gaussian Splatting. It addresses key performance bottlenecks and compatibility issues whilst improving image fidelity and metadata clarity. The resulting script is portable, maintainable, and well suited for scalable 3D dataset generation workflows.