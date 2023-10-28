import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "./"))
sys.path.append(str(Path(__file__).parent / "bpy/"))


import os
from collections import Counter
import tempfile
from itertools import product
import math
import json
from blicket import BlicketView
import utils
import random
import bpy
import numpy as np
import argparse


HOME = str(Path(__file__).parent / ".")

# Load the property file
with open(HOME + "/data/properties.json", "r") as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties["colors"].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties["materials"].items()]
    object_mapping = [(v, k) for k, v in properties["shapes"].items()]
    size_mapping = list(properties["sizes"].items())

shape_color_material_combos = []
with open(HOME + "/data/all.json", "r") as f:
    combos_json = json.load(f)
    for key in combos_json:
        shape_color_material_combos.extend(
            list(
                product(
                    [key], combos_json[key]["colors"], combos_json[key]["materials"]
                )
            )
        )


def generate_scene(light_state, object_list, scene_struct):
    if light_state != "no":
        if light_state == "on":
            utils.turn_on_blicket("BlicketLight", 1)
        else:
            utils.turn_on_blicket("BlicketLight", 0)

    # This will give ground-truth information about the scene and its objects
    scene_struct["light_state"] = light_state
    scene_struct["objects"] = []
    camera = bpy.data.objects["Camera"]
    # objects, blender_objects = add_objects(scene_struct, object_list, args, camera)
    blender_objects = add_objects(scene_struct, object_list, camera)

    return blender_objects


def warmup():
    for idx in range(len(shape_color_material_combos)):
        obj_name_out, color_name, mat_name_out = shape_color_material_combos[idx]
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
        mat_name = [k for k, v in material_mapping if v == mat_name_out][0]
        # Actually add the object to the scene
        # utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = utils.add_object(
            HOME + "/data/shapes",
            (obj_name, mat_name, color_name),
            rgba,
            0,
            (0, 0),
            theta=0,
        )


def add_objects(scene_struct, object_list, camera):
    """
    Add objects to the current blender scene, object attributes set according to object_list
    """

    for obj in bpy.data.objects:
        if obj.name.startswith("obj_"):
            obj.hide_render = True
            # obj.hide_viewport = True

    positions = []
    blender_objects = []
    for idx in object_list:
        # Choose a random size
        size_name, r = random.choice(size_mapping)

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > 50:  # max_retries
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_objects(scene_struct, object_list, camera)
            x = random.uniform(-2.4, 2.4)
            y = random.uniform(-2.4, 2.4)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for xx, yy, rr in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < 0.1:  # min_dist
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < 0.02:  # args.margin
                        # print("BROKEN MARGIN!")
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose color, shape, and mat
        obj_name_out, color_name, mat_name_out = shape_color_material_combos[idx]
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
        mat_name = [k for k, v in material_mapping if v == mat_name_out][0]

        # For cube, adjust the size a bit
        if obj_name == "Cube":
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        # utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = utils.add_object(
            HOME + "/data/shapes",
            (obj_name, mat_name, color_name),
            rgba,
            r,
            (x, y),
            theta=theta,
        )
        blender_objects.append(obj)
        positions.append((x, y, r))

    return blender_objects
