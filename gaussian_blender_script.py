import argparse
import json
import math
import os
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Define import functions for different file formats
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

# Set to True for verbose logging
VERBOSE = True

def log(message):
    "Print log message if verbose is enabled."
    if VERBOSE:
        print(f"[INFO] {message}")


def reset_scene() -> None:
    "Resets the scene to a clean state."
    log("Resetting scene")
    # Delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
    
    reset_cameras()


def reset_cameras() -> None:
    "Resets the cameras in the scene to a single default camera."
    log("Resetting cameras")
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'Camera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    bpy.context.scene.camera = new_camera


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    "Returns all root objects in the scene."
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    "Returns all meshes in the scene."
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    "Returns the bounding box of the scene."
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> None:
    "Normalises the scene by scaling and translating it to fit in a unit cube centred at the origin."
    log("Normalised Scene")
    if len(list(get_scene_root_objects())) > 1:
        # Create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # Parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty and obj.type != "CAMERA":
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        if obj.type != "CAMERA":
            obj.scale = obj.scale * scale

    # Apply scale to matrix_world
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        if obj.type != "CAMERA":
            obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    log("Scene normalised")


def load_object(object_path: str) -> bool:
    "Loads a model with a supported file extension into the scene."
    log(f"Loading object: {object_path}")
    
    if not os.path.exists(object_path):
        log(f"Error: File not found: {object_path}")
        return False
        
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        log(f"Error: Unsupported file type: {object_path}")
        return False

    try:
        if file_extension == "usdz":
            dirname = os.path.dirname(os.path.realpath(__file__))
            usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
            bpy.ops.preferences.addon_install(filepath=usdz_package)
            addon_name = "io_scene_usdz"
            bpy.ops.preferences.addon_enable(module=addon_name)
            from io_scene_usdz.import_usdz import import_usdz
            import_usdz(bpy.context, filepath=object_path, materials=True, animations=True)
            return True

        # Load from existing import functions
        if file_extension in IMPORT_FUNCTIONS:
            import_function = IMPORT_FUNCTIONS[file_extension]
            
            if file_extension == "blend":
                import_function(directory=object_path, link=False)
            elif file_extension in {"glb", "gltf"}:
                import_function(filepath=object_path, merge_vertices=True)
            else:
                import_function(filepath=object_path)
                
            # Check if any objects were loaded
            meshes = list(get_scene_meshes())
            if not meshes:
                log("Warning: No mesh objects were imported")
                
            log(f"Successfully loaded object with {len(meshes)} meshes")
            return True
        else:
            log(f"Error: Unsupported file extension: {file_extension}")
            return False
    except Exception as e:
        log(f"Error loading object: {str(e)}")
        return False


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: float = 1.0,
):
    "Creates a light object."
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    light_data.color = color
    
    # Set size for area lights
    if light_type == "AREA":
        light_data.size = size
        
    return light_object


def create_optimal_lighting() -> Dict[str, bpy.types.Object]:
    "Creates optimal lighting for photogrammetry captures with GREATLY enhanced brightness."
    log("Setting up enhanced sun lighting")
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light (primary light source) - GREATLY ENHANCED BRIGHTNESS
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),  # 45 degrees
        energy=20.0,  
        specular_factor=1.0,
        use_shadow=True,
        color=(1.0, 0.95, 0.9),  # Slightly warm
    )

    # Create fill light (reduces harsh shadows) - GREATLY ENHANCED BRIGHTNESS
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),  # Opposite side from key light
        energy=12.0,  
        specular_factor=0.8,
        use_shadow=False,
        color=(0.9, 0.95, 1.0),
    )

    # Create rim light (highlights edges) - GREATLY ENHANCED BRIGHTNESS
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=16.0,  
        specular_factor=1.0,
        use_shadow=False,
    )

    # Create bottom light (reduces completely black areas) - GREATLY ENHANCED BRIGHTNESS
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),  # From below
        energy=6.0, 
        specular_factor=0.5,
        use_shadow=False,
    )
    
    # Add a new ambient fill light to brighten dark areas
    ambient_light = _create_light(
        name="Ambient_Light",
        light_type="SUN", 
        location=(0, 0, 0),
        rotation=(0, 0, 0),  # From directly above
        energy=5.0,  # Ambient light
        specular_factor=0.3,
        use_shadow=False,
        color=(1.0, 1.0, 1.0),  # Pure white for fill
    )
    
    # Set world background brighter (ambient light)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    
    # Remove existing nodes
    for node in world_nodes:
        world_nodes.remove(node)
    
    # Create new background with BRIGHTER ambient
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1.0)  # DOUBLED brightness for ambient
    background.inputs['Strength'].default_value = 1.5  # Background light strength
    
    # Add world output
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    log("Enhanced sun lighting setup complete")
    return {
        "key_light": key_light,
        "fill_light": fill_light,
        "rim_light": rim_light,
        "bottom_light": bottom_light,
        "ambient_light": ambient_light,
    }


def create_area_lighting() -> Dict[str, bpy.types.Object]:
    "Creates optimal lighting for photogrammetry captures with GREATLY enhanced brightness."
    log("Setting up enhanced area lighting")
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light (primary light source) - GREATLY ENHANCED BRIGHTNESS
    key_light = _create_light(
        name="Key_Light",
        light_type="AREA",
        location=(1.5, -1.5, 2.0),
        rotation=(0.5, 0.2, 1.0),
        energy=20.0,  
        specular_factor=1.0,
        use_shadow=True,
        color=(1.0, 0.95, 0.9), 
        size=3.0,  # Set size for softer light
    )

    # Create fill light (reduces harsh shadows) - GREATLY ENHANCED BRIGHTNESS
    fill_light = _create_light(
        name="Fill_Light",
        light_type="AREA",
        location=(-1.5, -1.0, 1.0),
        rotation=(0.5, -0.2, -1.0),
        energy=10.0,  
        specular_factor=0.8,
        use_shadow=False,
        color=(0.9, 0.95, 1.0),  # Slightly cool
        size=3.0,
    )

    # Create rim light (highlights edges) - GREATLY ENHANCED BRIGHTNESS
    rim_light = _create_light(
        name="Rim_Light",
        light_type="AREA",
        location=(0.0, 2.0, 1.5),
        rotation=(-0.4, 0.0, 0.0),
        energy=14.0, 
        specular_factor=1.0,
        use_shadow=False,
        color=(1.0, 1.0, 1.0),
        size=3.0,
    )

    # Create ambient light (general fill) - GREATLY ENHANCED BRIGHTNESS
    ambient_light = _create_light(
        name="Ambient_Light",
        light_type="AREA",
        location=(0.0, 0.0, 3.0),
        rotation=(0.0, 0.0, 0.0),
        energy=8.0,
        specular_factor=0.5,
        use_shadow=False,
        color=(1.0, 1.0, 1.0),
        size=8.0,
    )
    
    # Add an additional front fill light for better illumination
    front_fill = _create_light(
        name="Front_Fill",
        light_type="AREA",
        location=(0.0, -2.0, 0.5),
        rotation=(0.3, 0.0, 0.0),
        energy=7.0,
        specular_factor=0.6,
        use_shadow=False,
        color=(1.0, 1.0, 1.0),
        size=4.0,
    )
    
    # Set world background much brighter (ambient light)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    
    # Remove existing nodes
    for node in world_nodes:
        world_nodes.remove(node)
    
    # Create new background with BRIGHTER ambient
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs['Colour'].default_value = (0.2, 0.2, 0.2, 1.0)  # DOUBLED brightness for ambient
    background.inputs['Strength'].default_value = 1.5  
    
    # Add world output
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    log("Enhanced area lighting setup complete")
    return {
        "key_light": key_light,
        "fill_light": fill_light,
        "rim_light": rim_light,
        "ambient_light": ambient_light,
        "front_fill": front_fill
    }


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    "Returns the 3x4 RT matrix from the given camera."
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
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


def position_camera_for_splat(index: int, total: int, radius: float = 2.0) -> bpy.types.Object:
    "Positions camera evenly around the object for optimal Gaussian Splat reconstruction."
    camera = bpy.data.objects["Camera"]
    
    # Calculate positions on a sphere for even distribution
    # Using enhanced Fibonacci sphere distribution for more even coverage
    golden_ratio = (1 + 5**0.5) / 2
    i = index
    
    # Distribution formula
    theta = 2 * math.pi * i / golden_ratio
    z = 1 - (2 * (i + 0.5) / total)
    phi = math.acos(z)
    
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    
    # Prevent cameras at exact poles
    if abs(z) > 0.95 * radius:
        z *= 0.95
    
    camera.location = Vector((x, y, z))

    # Point camera at the center
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    
    log(f"Camera {index} positioned at ({x:.2f}, {y:.2f}, {z:.2f})")
    return camera


def ensure_directory(path: str) -> None:
    "Ensures that the directory exists."
    try:
        os.makedirs(path, exist_ok=True)
        log(f"Directory ensured: {path}")
    except Exception as e:
        log(f"Error creating directory {path}: {str(e)}")


def setup_camera(lens: float = 35) -> None:
    "Sets up the camera with optimal parameters for Gaussian Splats."
    log("Setting up camera")
    cam = bpy.context.scene.objects["Camera"]
    
    # Set camera parameters - 35mm is generally good for photogrammetry
    cam.data.lens = lens
    cam.data.sensor_width = 32
    
    # Remove any existing constraints first
    for constraint in cam.constraints:
        cam.constraints.remove(constraint)
        
    # Set up camera constraints to track the center
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    # Create empty target at the center
    # First check if target already exists
    if "Target" in bpy.data.objects:
        target = bpy.data.objects["Target"]
    else:
        empty = bpy.data.objects.new("Target", None)
        bpy.context.scene.collection.objects.link(empty)
        target = empty
    
    cam_constraint.target = target
    log(f"Camera setup complete with {lens}mm lens")


def setup_render_settings(resolution: int = 512, engine: str = "CYCLES", samples: int = 64) -> None:
    "Sets up the render settings optimised for Gaussian Splats."
    log(f"Setting up render settings: {engine}, {resolution}x{resolution}, {samples} samples")
    render = bpy.context.scene.render
    
    # Check if using an older engine name and replace with compatible one
    if engine == "BLENDER_EEVEE":
        engine = "BLENDER_EEVEE_NEXT"
        log("Changed engine from BLENDER_EEVEE to BLENDER_EEVEE_NEXT")
    
    # Set render engine and resolution
    try:
        render.engine = engine
    except Exception as e:
        log(f"Error setting render engine: {str(e)}")
        available_engines = [e.bl_idname for e in bpy.types.RenderEngine.__subclasses__()]
        log(f"Available engines: {', '.join(available_engines)}")
        log("Falling back to CYCLES")
        render.engine = "CYCLES"
    
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    
    # Image settings for optimal output
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.film_transparent = True  # Transparent background
    
    # IMPORTANT: Disable compositor to prevent the png.47 folder issue
    bpy.context.scene.use_nodes = False
    
    # EEVEE settings for optimal lighting
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
                # Don't try to set gtao_factor if it doesn't exist
            if hasattr(eevee, 'use_ssr'):
                eevee.use_ssr = True
                if hasattr(eevee, 'ssr_quality'):
                    eevee.ssr_quality = 0.5
            # Only set these if they exist
            if hasattr(eevee, 'diffuse_bounces'):
                eevee.diffuse_bounces = 3
            if hasattr(eevee, 'glossy_bounces'):
                eevee.glossy_bounces = 3
    
    # Cycles specific settings for speed/quality balance
    if engine == "CYCLES":
        cycles = bpy.context.scene.cycles
        cycles.device = "GPU"
        cycles.samples = samples
        # Safely check these attributes to avoid similar errors
        if hasattr(cycles, 'diffuse_bounces'):
            cycles.diffuse_bounces = 3
        if hasattr(cycles, 'glossy_bounces'):
            cycles.glossy_bounces = 3
        if hasattr(cycles, 'transparent_max_bounces'):
            cycles.transparent_max_bounces = 3
        if hasattr(cycles, 'transmission_bounces'):
            cycles.transmission_bounces = 3
        cycles.filter_width = 0.01
        if hasattr(cycles, 'use_denoising'):
            cycles.use_denoising = True
        
        # Setup GPU computing if available
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
    
    log("Render settings configured")


def capture_metadata(object_path: str) -> Dict[str, Any]:
    "Captures minimal metadata needed for Gaussian Splats."
    bbox_min, bbox_max = scene_bbox()
    return {
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "object_path": object_path,
        "mesh_count": sum(1 for _ in get_scene_meshes()),
        "rendering_settings": {
            "engine": bpy.context.scene.render.engine,
            "resolution": bpy.context.scene.render.resolution_x
        }
    }


def render_for_gaussian_splats(
    object_path: str,
    output_dir: str,
    num_renders: int = 36,
    engine: str = "BLENDER_EEVEE_NEXT",
    resolution: int = 2048, 
    use_area_lights: bool = True,  # Area lights for better quality
    separate_matrices_dir: bool = True,  # Store matrices in a separate directory
) -> None:
    "Renders images optimised for Gaussian Splat generation."
    log(f"Starting Gaussian Splat rendering: {object_path} -> {output_dir}")
    
    ensure_directory(output_dir)
    
    # Create a separate directory for matrices if requested
    matrices_dir = os.path.join(output_dir, "matrices") if separate_matrices_dir else output_dir
    if separate_matrices_dir:
        ensure_directory(matrices_dir)
    
    # Reset scene and load object
    reset_scene()
    if not load_object(object_path):
        log("Failed to load object")
        return
    
    normalize_scene()
    
    setup_camera()
    
    # Setup lighting optimised for object visualisation with ENHANCED brightness
    if use_area_lights:
        create_area_lighting()  # Uses much brighter area lights
    else:
        create_optimal_lighting()
    
    # Configure render settings - IMPORTANT: This disables compositor
    setup_render_settings(
        resolution=resolution, 
        engine=engine,
        samples=64 if engine == "CYCLES" else 1
    )
    
    # Disable compositing to prevent Blender from creating unwanted temporary folders (like "png.47")
    bpy.context.scene.use_nodes = False
    
    # Save camera positions for colmap compatibility
    camera_positions = []
    
    # Render images with evenly distributed camera positions
    for i in range(num_renders):
        try:
            # Position the camera
            camera = position_camera_for_splat(i, num_renders)
            
            # Render the image - make sure direct path is set
            render_path = os.path.join(output_dir, f"{i:03d}.png")
            bpy.context.scene.render.filepath = render_path
            log(f"Rendering image {i+1}/{num_renders} to {render_path}")
            bpy.ops.render.render(write_still=True)
            
            # Get and save camera matrix
            rt_matrix = get_3x4_RT_matrix_from_blender(camera)
            rt_matrix_path = os.path.join(matrices_dir, f"{i:03d}.npy")
            np.save(rt_matrix_path, rt_matrix)
            
            # Store camera position
            camera_positions.append({
                "id": i,
                "position": list(camera.location),
                "rotation": [camera.rotation_euler.x, camera.rotation_euler.y, camera.rotation_euler.z],
                "image_file": f"{i:03d}.png",
                "matrix_file": f"matrices/{i:03d}.npy" if separate_matrices_dir else f"{i:03d}.npy"
            })
            log(f"Completed render {i+1}/{num_renders}")
        except Exception as e:
            log(f"Error during render {i+1}: {str(e)}")
    
    # Save camera positions in a format compatible with reconstruction tools
    try:
        camera_file = os.path.join(output_dir, "cameras.json")
        with open(camera_file, "w", encoding="utf-8") as f:
            json.dump({
                "cameras": camera_positions,
                "metadata": capture_metadata(object_path)
            }, f, indent=2)
        log(f"Camera data saved to {camera_file}")
    except Exception as e:
        log(f"Error saving camera data: {str(e)}")
    
    # Save a simple README
    try:
        readme_file = os.path.join(output_dir, "README.txt")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(f"Gaussian Splat rendering of {object_path}\n")
            f.write(f"Generated {num_renders} images at {resolution}x{resolution} resolution\n")
            f.write("Camera matrices are saved as .npy files\n")
            if separate_matrices_dir:
                f.write("Camera matrices are organised in the 'matrices' subdirectory\n")
            f.write("Use cameras.json for reconstruction software\n")
        log(f"README saved to {readme_file}")
    except Exception as e:
        log(f"Error saving README: {str(e)}")
    
    log("Gaussian Splat rendering complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render images for Gaussian Splat creation")
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the 3D object file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where renders will be saved"
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=36,
        help="Number of images to render (36 recommended for Gaussian Splats)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["CYCLES", "BLENDER_EEVEE_NEXT"],
        default="BLENDER_EEVEE_NEXT",
        help="Render engine to use (EEVEE is faster, CYCLES produces better quality)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2048,
        help="Resolution for rendered images (width = height)"
    )
    parser.add_argument(
        "--use_area_lights",
        action="store_true",
        default=True,  # Default to using area lights
        help="Use area lights instead of sun lights for better quality"
    )
    parser.add_argument(
        "--separate_matrices_dir",
        action="store_true",
        default=True,  # Default to storing matrices in separate directory
        help="Store camera matrices in a separate 'matrices' subdirectory"
    )
    
    # Parse arguments from command line
    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
        args = parser.parse_args(argv)
        log("Using command line arguments")
    except (ValueError, SystemExit):
        # If running from within Blender's Script Editor
        # Set default values for testing
        class Args:
            def __init__(self):
                self.object_path = "./models/example_model.glb" # Change to your 3D asset path
                self.output_dir = "./output" # Change to your output path
                self.num_renders = 36
                self.engine = "BLENDER_EEVEE_NEXT"
                self.resolution = 2048
                self.use_area_lights = True
                self.separate_matrices_dir = True
        args = Args()
        log("Running with default arguments. Please set paths in the script.")
    
    # Run the rendering
    render_for_gaussian_splats(
        object_path=args.object_path,
        output_dir=args.output_dir,
        num_renders=args.num_renders,
        engine=args.engine,
        resolution=args.resolution,
        use_area_lights=args.use_area_lights,
        separate_matrices_dir=args.separate_matrices_dir
    )