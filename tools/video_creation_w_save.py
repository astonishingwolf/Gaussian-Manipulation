import numpy as np
import bpy
import sys
import os
import math
import json
import mathutils
from mathutils import Vector
import gc # Add this at the top of your script

# Force stdout flush for print statements in Blender
sys.stdout.flush()
sys.stderr.flush()

def SmoothenObject(obj):
    """Apply smooth shading to an object."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()


def AddProceduralMaterial(obj):
    """Create and assign a procedural material with noise, bump, and color ramp.
    
    Compatible with Blender 4.2+ Principled BSDF changes.
    """
    mat_name = "Procedural_Texture"
    
    # Reuse existing material or create new one
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create shader nodes
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (400, 0)
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)
    node_bsdf.inputs['Roughness'].default_value = 0.4
    
    # Handle Blender 4.2+ specular property name change
    if 'Specular IOR Level' in node_bsdf.inputs:
        node_bsdf.inputs['Specular IOR Level'].default_value = 0.5
    elif 'Specular' in node_bsdf.inputs:
        node_bsdf.inputs['Specular'].default_value = 0.5

    # Texture coordinate node for 3D texturing
    node_coords = nodes.new(type='ShaderNodeTexCoord')
    node_coords.location = (-1000, 0)
    
    # Noise texture node
    node_noise = nodes.new(type='ShaderNodeTexNoise')
    node_noise.location = (-800, 0)
    node_noise.inputs['Scale'].default_value = 8.0
    node_noise.inputs['Detail'].default_value = 5.0
    node_noise.inputs['Distortion'].default_value = 0.2
    
    # Color ramp for gradient mapping
    node_ramp = nodes.new(type='ShaderNodeValToRGB')
    node_ramp.location = (-500, 200)
    node_ramp.color_ramp.elements[0].position = 0.3
    node_ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.3, 1)  # Dark blue
    node_ramp.color_ramp.elements[1].position = 0.7
    node_ramp.color_ramp.elements[1].color = (0.2, 0.8, 1.0, 1)  # Light cyan
    
    # Bump node for surface relief
    node_bump = nodes.new(type='ShaderNodeBump')
    node_bump.location = (-200, -150)
    node_bump.inputs['Strength'].default_value = 0.3

    # Connect nodes
    links.new(node_coords.outputs['Object'], node_noise.inputs['Vector'])
    links.new(node_noise.outputs['Fac'], node_ramp.inputs['Fac'])
    links.new(node_ramp.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_noise.outputs['Fac'], node_bump.inputs['Height'])
    links.new(node_bump.outputs['Normal'], node_bsdf.inputs['Normal'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def CreateObject(verts, faces):
    """Create a mesh object from vertices and faces."""
    mesh = bpy.data.meshes.new("ArticulatedMesh")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    # Link mesh object to scene
    mesh_object = bpy.data.objects.new("ArticulatedObject", mesh)
    bpy.context.collection.objects.link(mesh_object)
    return mesh_object


def CreateArmature(joints, bones):
    """Create an armature with bones based on joint positions."""
    armature_data = bpy.data.armatures.new("Armature")
    armature_obj = bpy.data.objects.new("ArmatureObject", armature_data)
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    for i in range(len(bones)):
        parent_index = bones[i, 0]
        child_index = bones[i, 1]
        bone = edit_bones.new(f"Bone_{child_index}")
        bone.head = joints[parent_index]
        bone.tail = joints[child_index]

    # Set bone parents
    for i in range(len(bones)):
        parent_index = bones[i, 0]
        child_index = bones[i, 1]
        bone = edit_bones.get(f"Bone_{child_index}")
        bone.parent = edit_bones.get(f"Bone_{parent_index}")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Lock root bone to prevent movement
    root_bone = armature_obj.pose.bones.get("Bone_0")
    if root_bone:
        root_bone.lock_location = (True, True, True)
        root_bone.lock_rotation = (True, True, True)
        root_bone.lock_scale = (True, True, True)

    armature_obj.show_in_front = True
    return armature_obj


def ApplySkinning(obj, armature_obj, skin_weights, skin_vert_inds, skin_joint_ind, skin_shape, bones):
    """Apply skinning weights to bind mesh to armature."""
    # Create vertex groups for each bone
    for j in range(skin_shape[1]):
        obj.vertex_groups.new(name=f"Bone_{j}")

    # Assign weights to vertex groups
    for i in range(len(skin_weights)):
        v_idx = int(skin_vert_inds[i])
        j_idx = int(skin_joint_ind[i])
        children = bones[bones[:, 0] == j_idx, 1]
        weight = float(skin_weights[i])
        
        if len(children) == 0:
            obj.vertex_groups[f"Bone_{j_idx}"].add([v_idx], weight, 'REPLACE')
        else:
            for child_index in children:
                obj.vertex_groups[f"Bone_{child_index}"].add([v_idx], weight, 'REPLACE')

    # Bind armature to mesh via modifier
    arm_mod = obj.modifiers.new("ArmatureMod", type='ARMATURE')
    arm_mod.object = armature_obj
    obj.parent = armature_obj


def ApplyVibration(armature, max_vibration_angle, vibration_time_period, total_duration):
    """Apply oscillating vibration animation to armature bones."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    pose_bones = armature.pose.bones

    # Generate random rotation axis for each bone
    random_axes = {bone.name: np.random.randn(3) for bone in pose_bones}
    
    for name in random_axes:
        random_axes[name] /= np.linalg.norm(random_axes[name])

    max_angle_rad = math.radians(max_vibration_angle)

    # Apply keyframed oscillation to each bone
    for frame in range(total_duration):
        bpy.context.scene.frame_set(frame)
        
        for bone_name, axis in random_axes.items():
            bone = pose_bones[bone_name]
            angle = max_angle_rad * math.sin(2 * math.pi * frame / vibration_time_period)
            rot_quat = mathutils.Quaternion(axis, angle)
            
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rot_quat
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    bpy.ops.object.mode_set(mode='OBJECT')


def OscillateCam(target_point, elevation_deg, radius, start_frame, time_period, n_oscillations, mode="single"):
    """Create and animate camera orbiting around target point."""
    # Delete existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Create new camera
    cam_data = bpy.data.cameras.new(name="Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Delete existing Sun lamps
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Create sun light parented to camera
    light_data = bpy.data.lights.new(name="CameraSun", type='SUN')
    light_data.energy = 5.0
    light_object = bpy.data.objects.new(name="CameraSun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.parent = cam

    # Setup camera orbit parameters
    target = Vector(target_point)
    elevation_rad = math.radians(elevation_deg)

    # Animate camera in circular orbit
    for f in range(start_frame, start_frame + time_period + 1):
        t = (f - start_frame) / time_period

        azimuth = 2 * math.pi * n_oscillations * t - (math.pi / 2)

        x = radius * math.cos(elevation_rad) * math.cos(azimuth)
        y = radius * math.cos(elevation_rad) * math.sin(azimuth)
        z = radius * math.sin(elevation_rad)

        cam.location = target + Vector((x, y, z))
        cam.keyframe_insert(data_path="location", frame=f)

        direction = target - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=f)

    print("[Camera] 360° orbit animation created.")
    return cam


def SaveCameraData(camera, save_path, start_frame, end_frame):
    """Extract and save camera intrinsics and extrinsics for each frame as JSON."""
    scene = bpy.context.scene
    render = scene.render
    os.makedirs(save_path, exist_ok=True)

    # Get render resolution
    W = render.resolution_x
    H = render.resolution_y
    sensor_width_mm = camera.data.sensor_width
    pixel_aspect = render.pixel_aspect_y / render.pixel_aspect_x

    total_frames = end_frame - start_frame + 1
    print(f"[→] Saving {total_frames} camera parameters...")
    
    for i, frame in enumerate(range(start_frame, end_frame + 1), 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        # Calculate camera intrinsics
        focal_length_mm = camera.data.lens
        focal_length_px = (focal_length_mm / sensor_width_mm) * W
        cx = W / 2.0
        cy = H / 2.0
        
        # Get camera world matrix (c2w in Blender/OpenGL convention)
        c2w_blender = camera.matrix_world
        
        # Convert from Blender/OpenGL to OpenCV convention
        # Blender/OpenGL: +Y up, -Z forward, +X right
        # OpenCV: +Y down, +Z forward, +X right
        # Conversion: flip Y and Z axes (multiply columns 1 and 2 by -1)
        c2w_opencv = mathutils.Matrix(c2w_blender)
        c2w_opencv[0][1] *= -1
        c2w_opencv[1][1] *= -1
        c2w_opencv[2][1] *= -1
        c2w_opencv[0][2] *= -1
        c2w_opencv[1][2] *= -1
        c2w_opencv[2][2] *= -1
        
        # Extract rotation and translation from c2w (OpenCV convention)
        R_c2w = c2w_opencv.to_3x3()
        t_c2w = c2w_opencv.translation
        
        # Build w2c (world-to-camera) transform
        R_w2c = R_c2w.transposed()
        t_w2c = -(R_w2c @ t_c2w)
        
        # For Nerfies format storage:
        # The reading code does: R = stored_orientation.T, then w2c[:3,:3] = R.T
        # So w2c rotation = (stored_orientation.T).T = stored_orientation
        # Therefore: stored_orientation should be R_w2c
        # For position: T = -stored_position @ stored_orientation (this is the bug in read code)
        # To work with existing reader: stored_position @ stored_orientation should give -T_w2c
        # stored_position @ R_w2c = -t_w2c
        # stored_position = -t_w2c @ R_w2c.T = -t_w2c @ R_c2w = t_c2w (camera position in world)
        
        orientation = [[R_w2c[i][j] for j in range(3)] for i in range(3)]
        position = [t_c2w[0], t_c2w[1], t_c2w[2]]

        # Build camera parameters dictionary
        cam_data = {
            "focal_length": focal_length_px,
            "image_size": [W, H],
            "orientation": orientation,
            "pixel_aspect_ratio": pixel_aspect,
            "position": position,
            "principal_point": [cx, cy],
            "radial_distortion": [0.0, 0.0, 0.0],
            "skew": 0.0,
            "tangential_distortion": [0.0, 0.0]
        }

        # Save to JSON file
        json_filename = os.path.join(save_path, f"0_{frame:05d}.json")
        with open(json_filename, 'w') as f:
            json.dump(cam_data, f, indent=4)
        
        if i % 50 == 0 or i == total_frames:
            print(f"  [{i}/{total_frames}] Saved camera {frame:05d}")
    
    print(f"[✔] Camera parameters saved to {save_path}")


def RenderVideo(save_path, start_frame, end_frame):
    """Render animation as video file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    bpy.context.scene.render.filepath = save_path
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.ops.render.render(animation=True)


def RenderPNGs(png_folder_path, start_frame, end_frame):
    """Render animation frames as PNG image sequence."""
    os.makedirs(png_folder_path, exist_ok=True)
    scene = bpy.context.scene

    # Set white background
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (1, 1, 1, 1)  # White
        bg_node.inputs['Strength'].default_value = 1.0
    
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 0

    total_frames = end_frame - start_frame + 1
    print(f"[→] Rendering {total_frames} RGB frames with white background...", flush=True)
    
    # Render each frame individually with 5-digit padding
    for i, frame in enumerate(range(start_frame, end_frame + 1), 1):
        scene.frame_set(frame)
        filename = f"0_{frame:05d}.png"
        scene.render.filepath = os.path.join(png_folder_path, filename)
        bpy.ops.render.render(write_still=True)
        
        # Force flush to ensure file is written
        bpy.context.view_layer.update()
        
        print(f"  [{i}/{total_frames}] Rendered frame {frame:05d}")

        # CRITICAL: Manually clear memory
        gc.collect()

    print(f"[✔] PNG sequence saved to {png_folder_path}")


def RenderMasks(mask_folder_path, obj, start_frame, end_frame):
    """Render object masks as PNG image sequence."""
    os.makedirs(mask_folder_path, exist_ok=True)
    scene = bpy.context.scene
    
    # Store original settings
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    original_color_mode = scene.render.image_settings.color_mode
    
    # Setup render settings for mask (use BLENDER_EEVEE_NEXT for Blender 4.2+)
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 0
    
    # Set black background for masks
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0, 0, 0, 1)  # Black
        bg_node.inputs['Strength'].default_value = 1.0
    
    # Store original material
    original_materials = [slot.material for slot in obj.material_slots]
    
    # Create white emission material for mask
    mask_mat = bpy.data.materials.new(name="MaskMaterial")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    nodes.clear()
    
    # Create emission shader (white)
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs['Color'].default_value = (1, 1, 1, 1)
    node_emission.inputs['Strength'].default_value = 1.0
    node_emission.location = (0, 0)
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (200, 0)
    
    mask_mat.node_tree.links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
    
    # Apply mask material to object
    if obj.data.materials:
        obj.data.materials[0] = mask_mat
    else:
        obj.data.materials.append(mask_mat)
    
    # Render frames individually with 5-digit padding
    total_frames = end_frame - start_frame + 1
    print(f"[→] Rendering {total_frames} mask frames...")
    
    for i, frame in enumerate(range(start_frame, end_frame + 1), 1):
        scene.frame_set(frame)
        filename = f"0_{frame:05d}.png"
        scene.render.filepath = os.path.join(mask_folder_path, filename)
        bpy.ops.render.render(write_still=True)
        
        # Force flush to ensure file is written
        bpy.context.view_layer.update()
        
        print(f"  [{i}/{total_frames}] Rendered mask {frame:05d}")
    
    # Restore original settings
    scene.render.engine = original_engine
    scene.render.film_transparent = original_film_transparent
    scene.render.image_settings.color_mode = original_color_mode
    
    # Restore original materials
    for i, mat in enumerate(original_materials):
        if i < len(obj.material_slots):
            obj.material_slots[i].material = mat
    
    # Clean up mask material
    bpy.data.materials.remove(mask_mat)
    
    print(f"[✔] Mask sequence saved to {mask_folder_path}")


def DeleteGeneratedObjects(mesh_name="ArticulatedObject", armature_name="ArmatureObject"):
    """Delete generated mesh and armature objects from scene."""
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Delete mesh object
    mesh_obj = bpy.data.objects.get(mesh_name)
    if mesh_obj:
        mesh_data = mesh_obj.data  # Store before deletion
        mesh_obj.select_set(True)
        bpy.ops.object.delete()
        print(f"Deleted mesh object: {mesh_name}")
        if mesh_data:
            bpy.data.meshes.remove(mesh_data, do_unlink=True)

    # Delete armature object
    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj:
        arm_data = arm_obj.data  # Store before deletion
        arm_obj.select_set(True)
        bpy.ops.object.delete()
        print(f"Deleted armature object: {armature_name}")
        if arm_data:
            bpy.data.armatures.remove(arm_data, do_unlink=True)

def export_deformed_objs(obj, output_folder, start_frame, end_frame):
    """Export deformed mesh as OBJ files for each frame."""
    os.makedirs(output_folder, exist_ok=True)
    scene = bpy.context.scene

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Apply Visual Geometry to Mesh
        bpy.ops.object.duplicate()  # duplicate original so we don’t lose animation
        dup = bpy.context.active_object
        bpy.ops.object.convert(target='MESH')

        # Export to OBJ
        filepath = os.path.join(output_folder, f"{frame:04d}.obj")
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            export_materials=False
        )
        bpy.data.objects.remove(dup, do_unlink=True)

        print(f"[✔] Exported frame {frame} → {filepath}")


def SamplePointsFromMesh(obj, num_samples=10000):
    """Sample points uniformly from mesh surface with normals."""
    import bmesh
    
    # Create a bmesh from the object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    # Calculate face areas for weighted sampling
    face_areas = np.array([f.calc_area() for f in bm.faces])
    total_area = face_areas.sum()
    face_probs = face_areas / total_area
    
    points = []
    normals = []
    
    # Sample points
    for _ in range(num_samples):
        # Select a face based on area-weighted probability
        face_idx = np.random.choice(len(bm.faces), p=face_probs)
        face = bm.faces[face_idx]
        
        # Sample random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        
        # Get vertices of the face
        verts = [v.co for v in face.verts]
        
        # Calculate point position using barycentric coordinates
        if len(verts) >= 3:
            v0 = Vector(verts[0])
            v1 = Vector(verts[1])
            v2 = Vector(verts[2])
            point = r1 * v0 + r2 * v1 + r3 * v2
            normal = face.normal
            
            points.append([point[0], point[1], point[2]])
            normals.append([normal[0], normal[1], normal[2]])
    
    bm.free()
    
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)


def SavePointCloud(points, normals, npy_path, ply_path):
    """Save point cloud as both .npy and .ply formats."""
    # Save as numpy array (xyz only)
    np.save(npy_path, points)
    print(f"[✔] Saved points.npy to {npy_path}")
    
    # Try to save as PLY with normals and RGB colors
    try:
        # Try importing plyfile
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            # If import fails, try adding user site-packages to path
            import site
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.insert(0, user_site)
            from plyfile import PlyData, PlyElement
        
        # Create structured array with xyz, normals, and RGB colors
        vertex_data = np.zeros(
            len(points),
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ]
        )
        
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['nx'] = normals[:, 0]
        vertex_data['ny'] = normals[:, 1]
        vertex_data['nz'] = normals[:, 2]
        
        # Set default gray color (128, 128, 128) for all points
        # You can modify this to use actual colors from your mesh/texture if available
        vertex_data['red'] = 128
        vertex_data['green'] = 128
        vertex_data['blue'] = 128
        
        # Create PLY element and save
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(ply_path)
        
        print(f"[✔] Saved points3d.ply to {ply_path}")
    except ImportError as e:
        print(f"[!] Warning: plyfile not available: {e}")
        print(f"[!] Skipping PLY export. Only NPY file saved.")
    except Exception as e:
        print(f"[!] Warning: Failed to save PLY file: {e}")


def GenerateDatasetJSON(output_path, start_frame, end_frame):
    """Generate dataset.json with frame IDs and train/val splits."""
    frame_ids = [f"0_{frame:05d}" for frame in range(start_frame, end_frame + 1)]
    
    # Split into train (even) and val (odd) frames
    train_ids = [f"0_{frame:05d}" for frame in range(start_frame, end_frame + 1) if frame % 2 == 0]
    val_ids = [f"0_{frame:05d}" for frame in range(start_frame, end_frame + 1) if frame % 2 == 1]
    
    dataset = {
        "count": len(frame_ids),
        "ids": frame_ids,
        "num_exemplars": len(frame_ids),
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"[✔] Dataset JSON saved to {output_path}")


def GenerateMetadataJSON(output_path, start_frame, end_frame):
    """Generate metadata.json with appearance_id, camera_id, and warp_id for each frame."""
    metadata = {}
    
    for frame in range(start_frame, end_frame + 1):
        frame_id = f"0_{frame:05d}"
        metadata[frame_id] = {
            "appearance_id": frame,
            "camera_id": 0,
            "warp_id": frame
        }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[✔] Metadata JSON saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    npz_data_folder = "/Users/tsaiiast/Desktop/MSCV/Learning3d/dg-mesh"
    data_root = "/Users/tsaiiast/Desktop/MSCV/Learning3d/dg-mesh/data/1931"
    video_output_folder = f"{data_root}/rgb/1x"
    cam_output_folder = f"{data_root}/camera"
    mask_output_folder = f"{data_root}/mask-tracking/1x/Annotations"
    dataset_json_path = f"{data_root}/dataset.json"
    metadata_json_path = f"{data_root}/metadata.json"
    
    mode = "360"
    
    os.makedirs(video_output_folder, exist_ok=True)
    os.makedirs(cam_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    start_time = 0
    end_time = 240
    indices = [1931]
    
    # Set render resolution
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100

    for index in indices:
        # Clear previous objects from scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        npz_path = f"{npz_data_folder}/{str(index).zfill(5)}.npz"
        data = np.load(npz_path, allow_pickle=True)
        mesh = data["arr_0"].item()

        obj = CreateObject(mesh["vertices"], mesh["faces"])
        SmoothenObject(obj)
        AddProceduralMaterial(obj)

        armature = CreateArmature(mesh["joints"], mesh["bones"])
        ApplySkinning(
            obj, 
            armature, 
            mesh["skinning_weights_value"],
            mesh["skinning_weights_row"],
            mesh["skinning_weights_col"],
            mesh["skinning_weights_shape"],
            mesh["bones"]
        )
        
        ApplyVibration(armature, 3, 20, end_time)
        cam_obj = OscillateCam([0.0, 0.0, 0.0], 20, 2.5, 0, end_time, 1, mode=mode)
        
        # Sample initial points from the mesh (before animation)
        bpy.context.scene.frame_set(0)
        points, normals = SamplePointsFromMesh(obj, num_samples=10000)
        SavePointCloud(points, normals, 
                      f"{data_root}/points.npy",
                      f"{data_root}/points3d.ply")
        
        # Render images directly to output folder
        RenderPNGs(video_output_folder, 0, end_time)
        
        # # Render masks directly to output folder
        RenderMasks(mask_output_folder, obj, 0, end_time)

        # # Save camera parameters directly to output folder
        SaveCameraData(cam_obj, cam_output_folder, 0, end_time)
        
        # # Generate dataset.json
        GenerateDatasetJSON(dataset_json_path, 0, end_time)
        
        # # Generate metadata.json
        GenerateMetadataJSON(metadata_json_path, 0, end_time)

        # # Cleanup scene
        DeleteGeneratedObjects()