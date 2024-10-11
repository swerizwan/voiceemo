import bpy
import os
import numpy as np
import sys

# Get filename and root directory from command line arguments
filename = str(sys.argv[-1])
root_dir = str(sys.argv[-2])

# List of blendshape names
model_bsList = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
                "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
                "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
                "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
                "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
                "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
                "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
                "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
                "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"]

# Get the 'face' object
obj = bpy.data.objects["face"]

# Set Blender scene settings
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'MATCAP'
bpy.context.scene.display.render_aa = 'FXAA'
bpy.context.scene.render.resolution_x = int(512)
bpy.context.scene.render.resolution_y = int(768)
bpy.context.scene.render.fps = 30
bpy.context.scene.render.image_settings.file_format = 'PNG'

def setup_blender_scene():
    """Set up Blender scene settings."""
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'MATCAP'
    bpy.context.scene.display.render_aa = 'FXAA'
    bpy.context.scene.render.resolution_x = int(512)
    bpy.context.scene.render.resolution_y = int(768)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.image_settings.file_format = 'PNG'

def set_camera_settings():
    """Set camera settings."""
    cam = bpy.data.objects['Camera']
    cam.scale = [2, 2, 2]
    bpy.context.scene.camera = cam

def render_frame(output_dir, filename, blendshape_data):
    """Render frames for each blendshape."""
    obj = bpy.data.objects["face"]
    model_bsList = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
                    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
                    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
                    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
                    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
                    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
                    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
                    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
                    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"]

    # Iterate over blendshapes
    for i, curr_bs in enumerate(blendshape_data):
        # Set blendshape values for the 'face' object
        for j, value in enumerate(curr_bs):
            obj.data.shape_keys.key_blocks[model_bsList[j]].value = value
        # Set output file path
        bpy.context.scene.render.filepath = os.path.join(output_dir, f'{filename}_{i}.png')
        # Render the frame
        bpy.ops.render.render(write_still=True)


# Set camera settings
cam = bpy.data.objects['Camera']
cam.scale = [2, 2, 2]
bpy.context.scene.camera = cam

# Output directory
output_dir = root_dir + filename
blendshape_path = root_dir + filename + '.npy'

result = []
bs = np.load(blendshape_path)

# Iterate over blendshapes
for i in range(bs.shape[0]):
    curr_bs = bs[i]
    # Set blendshape values for the 'face' object
    for j in range(52):
        obj.data.shape_keys.key_blocks[model_bsList[j]].value = curr_bs[j]
    # Set output file path
    bpy.context.scene.render.filepath = os.path.join(output_dir, '{}.png'.format(i))
    # Render the frame
    bpy.ops.render.render(write_still=True)
