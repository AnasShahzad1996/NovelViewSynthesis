# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, argparse
import json
import bpy
import mathutils
from mathutils import Vector
import numpy as np


DEBUG = False
VOXEL_NUMS = 512
# VIEWS = 200
# RESOLUTION = 800
RESULTS_PATH = 'rgb'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
# RANDOM_VIEWS = True
# UPPER_VIEWS = False
CIRCLE_FIXED_START = (-0.4,0,0)
CIRCLE_FIXED_END = (-.2,0,0)


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('output', type=str, help='path where files will be saved')
parser.add_argument('--random_views', action='store_true', default=False)
parser.add_argument('--resolution', type=int, default=800)
parser.add_argument('--views', type=int, default=200)
parser.add_argument('--prefix', type=int, default=0)
parser.add_argument('--cam_scale', type=int, default=1)
parser.add_argument('--i_offset', type=int, default=1)
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

argv = sys.argv
argv = argv[argv.index("--") + 1:]
args = parser.parse_args(argv)

RANDOM_VIEWS = args.random_views
RESOLUTION = args.resolution
VIEWS = args.views
PREFIX = args.prefix

np.random.seed(args.seed)  # fixed seed

homedir = args.output
fp = bpy.path.abspath(f"{homedir}/{RESULTS_PATH}")

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)
if not os.path.exists(os.path.join(homedir, "pose")):
    os.mkdir(os.path.join(homedir, "pose"))

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

# bounding box
for object in bpy.context.scene.objects:
    if object.name == 'BoundingBoxBox':
    # if 'Camera' not in obj.name:
        print(object.bound_box)
        bbox = [Vector(corner) for corner in object.bound_box]
        bbox = [min([bb[i] for bb in bbox]) for i in range(3)] + \
               [max([bb[i] for bb in bbox]) for i in range(3)]
        voxel_size = ((bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]) / VOXEL_NUMS) ** (1/3)
        print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]), 
            file=open(os.path.join(homedir, 'bbox.txt'), 'w'))

        print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]))

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (0, 5.0 * args.cam_scale, 0.5)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
rotation_mode = 'XYZ'

out_data['frames'] = []

b_empty.rotation_euler = CIRCLE_FIXED_START
b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff


rots = []

for i in range(0 + args.start, VIEWS + args.start):
    scene.render.filepath = os.path.join(fp, '{}_{:04d}'.format(PREFIX, i))

    bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': scene.render.filepath,
        'rotation': radians(stepsize),
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    with open(os.path.join(homedir, "pose", '{}_{:04d}.txt'.format(PREFIX, i)), 'w') as fo:
        for ii, pose in enumerate(frame_data['transform_matrix']):
            print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p) 
                            for j, p in enumerate(pose)]), 
                file=fo)
    out_data['frames'].append(frame_data)

    if RANDOM_VIEWS:
        r_inner = np.random.uniform(0, 2*np.pi)
    else:
        r_inner = radians(stepsize*i)

    b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(r_inner)+1)/2 * vertical_diff

    if RANDOM_VIEWS:
        b_empty.rotation_euler[2] = np.random.uniform(0, 2*np.pi)
    else:
        b_empty.rotation_euler[2] += radians(2*stepsize)


if not DEBUG:
    with open(os.path.join(homedir, 'transforms.json'), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

# save camera data
H, W = RESOLUTION, RESOLUTION
f = .5 * W /np.tan(.5 * float(out_data['camera_angle_x']))
cx = cy = W // 2

# write intrinsics
with open(os.path.join(homedir, 'intrinsics.txt'), 'w') as fi:
    print("{} {} {} 0.".format(f, cx, cy), file=fi)
    print("0. 0. 0.", file=fi)
    print("0.", file=fi)
    print("1.", file=fi)
    print("{} {}".format(H, W), file=fi)