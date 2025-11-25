import numpy as np
import bpy
import bmesh
import sys
import os
import math
import random
import mathutils
from mathutils import Vector, Euler
import time

# Get absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Add this directory to sys.path if it's not already there
if script_dir not in sys.path:
    sys.path.append(script_dir)

def SmoothenObject(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

def CreateObject(verts, faces):
    # === CREATE MESH OBJECT ===
    # Create a new mesh data block in Blender to hold the geometry
    mesh = bpy.data.meshes.new("ArticulatedMesh")

    # Fill the mesh data with vertex and face info
    mesh.from_pydata(verts.tolist(), [], faces.tolist())  # Second argument is edge list (empty here)
    mesh.update()  # Let Blender recalculate normals and topology

    # Create an object to hold the mesh and link it to the scene
    mesh_object = bpy.data.objects.new("ArticulatedObject", mesh)
    bpy.context.collection.objects.link(mesh_object)  # Add mesh object to the active Blender scene
    
    return mesh_object

def CreateArmature(joints, bones):
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

    # === NEW: lock root bone so it never moves ===
    root_bone = armature_obj.pose.bones.get("Bone_0")
    if root_bone:
        root_bone.lock_location = (True, True, True)
        root_bone.lock_rotation = (True, True, True)
        root_bone.lock_scale = (True, True, True)

    armature_obj.show_in_front = True
    return armature_obj


def ApplySkinning(obj, armature_obj, skin_weights, skin_vert_inds, skin_joint_ind, skin_shape, bones):
    # Create a vertex group for each bone (one per joint)
    for j in range(skin_shape[1]):
        obj.vertex_groups.new(name=f"Bone_{j}")

    # Create a mapping for 
    # Fill in the weights for each vertex group using sparse indices
    for i in range(len(skin_weights)):
        v_idx = int(skin_vert_inds[i])         # Vertex index
        j_idx = int(skin_joint_ind[i])         # Joint (bone) index
        children = bones[bones[:, 0] == j_idx, 1] # All children to this joint
        weight = float(skin_weights[i])        # Weight value
        
        if len(children) == 0:
            obj.vertex_groups[f"Bone_{j_idx}"].add([v_idx], weight, 'REPLACE')
        else:
            for child_index in children:
                # Add the weight to the appropriate vertex group
                obj.vertex_groups[f"Bone_{child_index}"].add([v_idx], weight, 'REPLACE')

    # === BIND ARMATURE TO MESH ===
    # Add an Armature modifier to the mesh object
    arm_mod = obj.modifiers.new("ArmatureMod", type='ARMATURE')
    arm_mod.object = armature_obj  # Assign the armature object to the modifier

    # Parent mesh to armature for transform inheritance (optional, good for organization)
    obj.parent = armature_obj

def ApplyVibration(armature, max_vibration_angle, vibration_time_period, total_duration):
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    pose_bones = armature.pose.bones

    # Assign a random axis per bone
    random_axes = {
        bone.name: np.random.randn(3)
        for bone in pose_bones
    }
    # Normalize all axes
    for name in random_axes:
        random_axes[name] /= np.linalg.norm(random_axes[name])

    max_angle_rad = math.radians(max_vibration_angle)

    for frame in range(total_duration):
        bpy.context.scene.frame_set(frame)
        
        print(len(random_axes.items()))
        for bone_name, axis in random_axes.items():
            
            bone = pose_bones[bone_name]
            
            # Compute oscillating angle
            angle = max_angle_rad * math.sin(2 * math.pi * frame / vibration_time_period)

            # Create a quaternion from axis-angle
            rot_quat = mathutils.Quaternion(axis, angle)
            
            # Reset to identity before applying rotation (optional but cleaner)
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rot_quat

            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    bpy.ops.object.mode_set(mode='OBJECT')

def OscillateCam(
    target_point, 
    elevation_deg, 
    radius, 
    start_frame, 
    time_period, 
    n_oscillations,
    mode="single"  # NEW ARGUMENT
):
    # === Delete existing cameras ===
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # === Create a new camera ===
    cam_data = bpy.data.cameras.new(name="Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Delete existing Sun lamps
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Create Sun light parented to camera
    light_data = bpy.data.lights.new(name="CameraSun", type='SUN')
    light_object = bpy.data.objects.new(name="CameraSun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.parent = cam

    # === Shared camera variables ===
    target = Vector(target_point)
    elevation_rad = math.radians(elevation_deg)

    # ============================================================
    # MODE 1: FIXED SINGLE VIEW
    # ============================================================
#    if mode == "single":
    azimuth = math.radians(45)   # Choose any angle you prefer

    x = radius * math.cos(elevation_rad) * math.cos(azimuth)
    y = radius * math.cos(elevation_rad) * math.sin(azimuth)
    z = radius * math.sin(elevation_rad)

    cam.location = target + Vector((x, y, z))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Insert only one keyframe
    cam.keyframe_insert(data_path="location", frame=start_frame)
    cam.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    print("[Camera] Single fixed view activated.")
    return cam

#    # ============================================================
#    # MODE 2: ORIGINAL OSCILLATING CAMERA
#    # ============================================================
#    for f in range(start_frame, start_frame + time_period + 1):
#        t = (f - start_frame) / time_period

#        azimuth = 2 * math.pi * n_oscillations * t - (math.pi / 2)

#        x = radius * math.cos(elevation_rad) * math.cos(azimuth)
#        y = radius * math.cos(elevation_rad) * math.sin(azimuth)
#        z = radius * math.sin(elevation_rad)

#        cam.location = target + Vector((x, y, z))
#        cam.keyframe_insert(data_path="location", frame=f)

#        direction = target - cam.location
#        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
#        cam.keyframe_insert(data_path="rotation_euler", frame=f)

#    print("[Camera] Oscillating 360° mode activated.")
    
    return cam


def RenderVideo(save_path, start_frame, end_frame):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set render output path (without extension)
    bpy.context.scene.render.filepath = save_path

    # Set frame range
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    # Render the animation
    bpy.ops.render.render(animation=True)

def RenderPNGs(png_folder_path, start_frame, end_frame):
    """
    Render animation frames as PNG images to the given folder.

    Args:
        png_folder_path (str): Folder where PNGs will be saved.
        start_frame (int): First frame to render.
        end_frame (int): Last frame to render.
    """
    # Ensure directory exists
    os.makedirs(png_folder_path, exist_ok=True)

    scene = bpy.context.scene

    # Set output path (Blender will append frame numbers automatically)
    scene.render.filepath = os.path.join(png_folder_path, "")

    # Set output format to PNG
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'  # Use 'RGB' if no transparency
    scene.render.image_settings.compression = 0      # 0 = lossless

    # Set frame range
    scene.frame_start = start_frame
    scene.frame_end = end_frame

    # Render the animation
    bpy.ops.render.render(animation=True)

    print(f"[✔] PNG sequence saved to {png_folder_path}")

def DeleteGeneratedObjects(mesh_name="ArticulatedObject", armature_name="ArmatureObject"):
    # Deselect all
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
    os.makedirs(output_folder, exist_ok=True)
    scene = bpy.context.scene

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        # Select object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Apply Visual Geometry to Mesh
        bpy.ops.object.duplicate()  # duplicate original so we don’t lose animation
        dup = bpy.context.active_object
        bpy.ops.object.convert(target='MESH')

        # Export to OBJ
        filepath = os.path.join(output_folder, f"{frame:04d}.obj")
        
        bpy.ops.wm.obj_export(filepath=filepath,
                              export_selected_objects=True,
                              export_materials=False)

        # Delete baked duplicate
        bpy.data.objects.remove(dup, do_unlink=True)

        print(f"[✔] Exported frame {frame} → {filepath}")

if __name__ == "__main__":
    npz_data_folder = "/Users/tsaiiast/Downloads/Custom_video"
    obj_output_folder = "/Users/tsaiiast/Downloads/Custom_video/meshes"
    video_output_folder = "/Users/tsaiiast/Downloads/Custom_video/videos_single"
    mode = "Single"
    
    os.makedirs(obj_output_folder, exist_ok=True)
    os.makedirs(video_output_folder, exist_ok=True)

    start_time = 0
    end_time = 240
    indices = [133]
    # len(os.listdir(data_folder))

    #(133 is lamp, 376 is spider)
    for index in indices: # Give range of object indices to process
        npz_path = f"{npz_data_folder}/{str(index).zfill(5)}.npz"
        # output_video_path = os.path.join(render_video_output_folder, f"{str(index).zfill(5)}.mp4")

        # Load npz data for the object
        data = np.load(npz_path, allow_pickle=True)
        mesh = data["arr_0"].item()

        # Create mesh
        obj = CreateObject(mesh["vertices"], mesh["faces"])
        SmoothenObject(obj)

        # Create armature and apply skinning
        armature = CreateArmature(mesh["joints"], mesh["bones"])
        ApplySkinning(obj, 
                    armature, 
                    mesh["skinning_weights_value"],
                    mesh["skinning_weights_row"],
                    mesh["skinning_weights_col"],
                    mesh["skinning_weights_shape"],
                    mesh["bones"]
                    )
        
        # Animate the armature
        ApplyVibration(armature, 3, 20, end_time)

        # Save obj file for each frame
        # export_deformed_objs(obj, obj_output_folder, start_time, end_time)

        # Animate the camera
#        OscillateCam([0.0, 0.0, 0.0], 20, 2.5, 0, end_time, 1, mode=mode)
        
        # Render the video
        video_output_folder = os.path.join(video_output_folder, f"{str(index).zfill(5)}")
        RenderPNGs(video_output_folder, 0, end_time)
        DeleteGeneratedObjects()