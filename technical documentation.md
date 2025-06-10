# Gaussian Blender Script: Technical Documentation

## Overview

During my research into 3D reconstruction techniques, I developed `gaussian_blender_script.py`, an 807-line Python script for generating high-quality training data for Gaussian Splatting models. This project began as a learning exercise to understand how photogrammetry works, but has evolved into a specialised tool that I have been using for my computer vision experiments.

The script automates the process of creating multiple camera views of 3D objects in Blender, which is essential for training Gaussian Splat models. Through trial and error, I have learnt that consistent camera placement and proper lighting are crucial for obtaining good reconstruction results.

**Reference Point**: My work builds upon and extends concepts from established 3D rendering pipelines, particularly drawing inspiration from the [AllenAI Objaverse-XL rendering scripts](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering). Whilst their `blender_script.py` focuses on general 3D object rendering for dataset creation, my implementation specialises in the specific requirements of Gaussian Splatting workflows, with enhanced camera positioning algorithms and photogrammetry-optimised lighting systems.

## What I Have Learnt and Built

### The Problem I Was Attempting to Solve

When I first began working with Gaussian Splatting, I quickly realised that creating good training datasets was surprisingly difficult. Whilst studying existing rendering pipelines such as those used in Objaverse-XL, I noticed they were optimised for general 3D visualisation rather than the specific needs of photogrammetric reconstruction. Manual camera placement in Blender was time-consuming and often resulted in uneven coverage of the object. I also struggled with:

- Obtaining consistent lighting across all views
- Ensuring the camera metadata was in the correct format for reconstruction software
- Making the script work on different computers (some had NVIDIA GPUs, others did not)
- Managing the large number of output files efficiently
- Achieving the photogrammetric quality needed for Gaussian Splatting (higher than general rendering)

### Key Features I Have Implemented

Through research and experimentation, building on established techniques from projects such as Objaverse-XL, I have developed several specialised features:

- **Intelligent Camera Placement**: I discovered the Fibonacci sphere algorithm through a computer graphics paper and implemented it to distribute cameras evenly around objects (improving upon simpler grid-based approaches)
- **Flexible Lighting Systems**: After testing different setups and studying how professional photogrammetry is conducted, I created two lighting configurations optimised for reconstruction quality
- **High-Resolution Output**: I increased the default resolution to 2048×2048 after finding that higher resolution significantly improves reconstruction quality compared to the 512px often used in general rendering
- **Cross-Platform GPU Support**: I have learnt how to detect different GPU types (NVIDIA CUDA versus Apple Metal) and automatically fall back to CPU rendering when needed
- **Robust File Import**: The script can handle many different 3D file formats, which I added as I encountered different model types during my experiments

## Technical Implementation

### Comparison with Existing Approaches

After studying the Objaverse-XL rendering pipeline, I identified several areas where I could specialise for Gaussian Splatting:

| **Aspect** | **General Rendering (e.g., Objaverse-XL)** | **My Gaussian Splat Implementation** |
|------------|---------------------------------------------|-------------------------------------|
| Camera Placement | Grid-based or random sampling | Fibonacci sphere distribution |
| Resolution | 512-1024px (storage efficient) | 2048px+ (reconstruction quality) |
| Lighting | Single or simple multi-light | Calibrated 5-point photographic setup |
| Output Format | Images only | Images + precise camera matrices |
| GPU Handling | Platform-specific | Auto-detection with fallbacks |

### File Import System

Building on the approach I observed in established rendering pipelines, I created a dictionary-based system that maps file extensions to the appropriate Blender import functions:

```python
IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    # ... more formats
}
```

This approach, inspired by modular design principles I observed in professional pipelines, made it straightforward to add support for new file types as I encountered them. I also learnt that some formats require special handling - for example, GLTF files work better when you merge duplicate vertices during import.

### Scene Preparation

Before rendering, I normalise all models to fit within a standard size. This was something I learnt through experience after obtaining inconsistent results with models of different scales:

```python
def normalise_scene() -> None:
    # Calculate how large the object is
    bbox_min, bbox_max = scene_bbox()
    # Scale it to fit in a unit cube
    scale = 1 / max(bbox_max - bbox_min)
    # Apply the scaling to all objects except cameras
    for obj in get_scene_root_objects():
        if obj.type != "CAMERA":
            obj.scale = obj.scale * scale
```

This ensures that whether I am rendering a tiny screw or a large building, the camera positioning and lighting work consistently - a crucial requirement for Gaussian Splatting that general rendering pipelines often overlook.

### Camera Placement Strategy

The most important part of the script is how it places cameras. Initially, I tried approaches similar to what I observed in general rendering pipelines (random positioning or simple grids), but this created uneven coverage with some areas having too many views and others having too few. After researching photogrammetry literature and computer graphics papers, I implemented the Fibonacci sphere algorithm:

```python
def position_camera_for_splat(index: int, total: int, radius: float = 2.0):
    # Using the golden ratio for even distribution
    golden_ratio = (1 + 5**0.5) / 2
    i = index
    
    # Mathematical formula for even spacing on a sphere
    theta = 2 * math.pi * i / golden_ratio
    z = 1 - (2 * (i + 0.5) / total)
    phi = math.acos(z)
    
    # Convert to 3D coordinates
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
```

This mathematical approach ensures that cameras are distributed evenly around the object, which I have found produces much better reconstruction results than the simpler placement methods used in general rendering applications.

### Lighting Systems

Obtaining good lighting was another major challenge. After studying professional photogrammetry setups and experimenting with various configurations, I developed two different lighting systems optimised for reconstruction quality rather than visual appeal:

#### Sun-Based Lighting
This uses five directional lights positioned strategically around the object. I have learnt that having multiple light sources prevents harsh shadows that can confuse the reconstruction algorithm.

#### Area-Based Lighting
This uses larger, softer light sources that create more natural-looking illumination. I have found this works particularly well for objects with complex surface details.

Both systems use enhanced brightness values that I calibrated through testing with different object types, going beyond the simpler lighting setups typically used in general 3D rendering.

### GPU Detection and Fallback

One of the most significant challenges I encountered was making the script work on different computers. Some had NVIDIA GPUs, others had Apple's Metal, and some only had CPU rendering available. I implemented an automatic detection system:

```python
if sys.platform == "darwin":  # macOS
    device_type = "METAL"
else:
    device_type = "CUDA"  # Default for Windows/Linux
    
try:
    # Try to configure GPU rendering
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.compute_device_type = device_type
    log(f"Successfully configured {device_type} rendering")
except Exception as e:
    log(f"GPU not available, falling back to CPU: {str(e)}")
    cycles.device = "CPU"
```

This approach means the script works on any computer, though it renders faster on machines with compatible GPUs - an improvement over platform-specific solutions I observed in some existing tools.

### Handling Different Blender Versions

I discovered that different versions of Blender have slightly different APIs, which caused crashes when running on older or newer versions. I solved this by adding safety checks:

```python
if engine == "BLENDER_EEVEE_NEXT" or engine == "BLENDER_EEVEE":
    if hasattr(bpy.context.scene, 'eevee'):
        eevee = bpy.context.scene.eevee
        # Only try to set properties that exist in this version
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 64
        if hasattr(eevee, 'use_gtao'):
            eevee.use_gtao = True
```

This defensive programming approach prevents crashes when running on different Blender installations.

### Camera Matrix Export

For Gaussian Splatting to work, the reconstruction software needs to know exactly where each camera was positioned. Unlike general rendering where this information is not always needed, I implemented a function to export this information in the standard computer vision format:

```python
def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Extract camera position and rotation as a 3x4 matrix"""
    # Get the camera's world position and rotation
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1 * R_world2bcam @ location
    
    # Combine into standard RT matrix format
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT
```

## Performance Improvements Made

Through iterative development, building on lessons learnt from studying existing rendering pipelines, I achieved several improvements over my initial version:

| **Aspect**          | **Before** | **After** | **Improvement** |
|---------------------|------------|-----------|-----------------|
| Code Length         | ~899 lines | 807 lines | 10% reduction (cleaner structure) |
| Function Count      | 25         | 13        | Simplified architecture |
| Output Resolution   | 512px      | 2048px    | 4× higher quality for reconstruction |
| Render Samples      | 32         | 64        | Better quality/noise ratio |
| Platform Support    | CUDA only  | Auto-detect | Works everywhere |
| File Output         | Verbose    | Compact   | Faster I/O and processing |

The most significant improvement was increasing the output resolution, which I found dramatically improves the quality of the final Gaussian Splat reconstruction compared to resolutions typically used in general rendering applications.

## How to Use the Script

### Basic Command Line Usage

```bash
blender --background --python gaussian_blender_script.py -- \
    --object_path "/path/to/model.glb" \
    --output_dir "/path/to/output" \
    --num_renders 36
```

### Available Parameters

| **Parameter** | **Type** | **Default** | **Purpose** |
|---------------|----------|-------------|-------------|
| `--object_path` | string | Required | Path to your 3D model file |
| `--output_dir` | string | Required | Where to save the rendered images |
| `--num_renders` | integer | 36 | Number of different camera angles to render |
| `--resolution` | integer | 2048 | Size of each rendered image (width and height) |
| `--engine` | string | BLENDER_EEVEE_NEXT | Which rendering engine to use |
| `--use_area_lights` | boolean | True | Whether to use soft area lights or directional sun lights |
| `--separate_matrices_dir` | boolean | True | Whether to save camera data in a separate folder |

### Output Structure

After running the script, you will obtain this file organisation:

```
output_dir/
├── 000.png          # First rendered image
├── 001.png          # Second rendered image
├── ...              # Additional images
├── cameras.json     # Camera position data (crucial for reconstruction)
├── README.txt       # Summary of what was rendered
└── matrices/        # Individual camera matrix files
    ├── 000.npy
    ├── 001.npy
    └── ...
```

## Challenges I Encountered and Solved

### Memory Management
Initially, the script would crash when rendering large numbers of images due to memory buildup. I learnt to clear Blender's cache between renders and manage GPU memory more carefully.

### Cross-Platform Compatibility
Getting the script to work on Windows, macOS, and Linux required learning about different GPU architectures and file path conventions - something I noticed was handled well in established pipelines such as Objaverse-XL.

### Blender Version Differences
Different versions of Blender have slightly different APIs, which required adding compatibility checks throughout the code.

### Photogrammetric Quality versus Speed
Balancing the high quality needed for Gaussian Splatting with reasonable rendering times required careful optimisation of render settings.

## Comparison with Existing Solutions

### How This Differs from General Rendering Pipelines

Whilst my script draws inspiration from established projects such as Objaverse-XL, it is specifically optimised for Gaussian Splatting:

**Objaverse-XL Strengths:**
- Massive scale processing
- Robust cloud deployment
- General-purpose 3D visualisation
- Efficient storage and bandwidth usage

**My Implementation Focus:**
- Photogrammetric quality optimisation
- Precise camera metadata for reconstruction
- Enhanced resolution for detail capture
- Specialised lighting for surface reconstruction
- Cross-platform compatibility for research environments

## Future Improvements I Would Like to Make

- **Adaptive Camera Placement**: Currently uses fixed sphere distribution, but could be improved to focus more cameras on complex areas of the object
- **Automatic Lighting Optimisation**: Could analyse the object's materials and adjust lighting accordingly
- **Progress Visualisation**: Add a preview mode to show camera positions before rendering
- **Batch Processing**: Support for processing multiple objects in sequence (similar to Objaverse-XL's approach)
- **Quality Validation**: Automatic checks to ensure rendered images meet reconstruction quality standards
- **Integration with Cloud Platforms**: Scale to handle larger datasets like the established pipelines

## What I Have Learnt from This Project

This project has taught me several important lessons:

1. **Building upon Existing Work**: Studying established pipelines such as Objaverse-XL provided valuable insights into robust system design
2. **Specialisation versus Generalisation**: Sometimes a specialised tool performs better than adapting a general-purpose solution
3. **Iterative Development**: Starting with a simple version and gradually adding features was much more effective than attempting to build everything at once
4. **Cross-Platform Considerations**: Testing on different operating systems and hardware configurations is crucial for robust software
5. **User Experience**: Adding helpful error messages and automatic fallbacks makes the difference between a tool that works and one that is actually usable
6. **Documentation**: Clear documentation (such as this file) is almost as important as the code itself

## Compatibility and Requirements

- **Blender Versions**: Tested on 3.0 through 4.0, with automatic compatibility detection
- **Operating Systems**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **GPU Support**: NVIDIA CUDA, Apple Metal, with automatic CPU fallback
- **Memory Requirements**: Minimum 8GB RAM recommended for high-resolution rendering
- **Storage**: Approximately 50MB per object (36 images at 2048×2048 resolution)

## Conclusion

The `gaussian_blender_script.py` represents my progression from manual dataset creation to an automated, specialised pipeline for Gaussian Splatting. Drawing inspiration from established systems such as Objaverse-XL, it addresses key challenges including inconsistent camera placement, insufficient image quality, and cross-platform compatibility through mathematical camera distribution and photogrammetry-optimised settings. This specialised approach achieves better reconstruction quality than general rendering solutions whilst providing a practical foundation for research-scale Gaussian Splatting experiments.