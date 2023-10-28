import gymnasium as gym
import os
import sys
import numpy as np
from PIL import Image
from gym import spaces
from pathlib import Path
from src.const import ALL_CONFIG_SIZE, MAX_TRIAL, UNIQUE_OBJ_CNT, CONTEXT_PER_SEQ, CONFIDENCE_THRES, EPS, IMG_WIDTH, IMG_HEIGHT, OBJ_MAX_BLICKET

sys.path.append(str(Path(__file__).parent / "./"))
sys.path.append(str(Path(__file__).parent / "../"))
sys.path.append(str(Path(__file__).parent / "render/bpy/"))

# import bpy
# from mathutils import Vector
# import time
# from render import utils
# from render.render_images import generate_scene, warmup


class SymbolicIVRE(gym.Wrapper):
    """Symbolic Version IVRE Wrapper"""

    def __init__(self, env):
        super().__init__(env)
        self.obs_shape = (OBJ_MAX_BLICKET + (MAX_TRIAL + 1) * UNIQUE_OBJ_CNT,)
        # (lasttrial, ans; last belief)
        self.action_shape = (2 * (UNIQUE_OBJ_CNT - 1),)
        # (belief; trial)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.action_shape, dtype=np.float32
        )
        self.env = env
        self.stepcntdict = {}
        self.reset()

    def step(self, action):
        action = action / 2.0 + 0.5
        action = np.clip(action, 0.0, 1.0)
        belief = np.zeros(self.env.config_size, dtype=np.float32)
        trial = np.zeros(self.env.config_size, dtype=np.float32)
        for index, prob in enumerate(action[: UNIQUE_OBJ_CNT - 1]):
            belief[self.index2obj[index]] = prob

        for index, prob in enumerate(action[UNIQUE_OBJ_CNT - 1:]):
            trial[self.index2obj[index]] = prob > 0.5

        # try:
        #     self.stepcntdict[self.env.stepcnt].append(int(trial.sum()))
        # except:
        #     self.stepcntdict[self.env.stepcnt] = []
        #     self.stepcntdict[self.env.stepcnt].append(int(trial.sum()))

        action_dict = {"belief": belief, "trial": trial}

        observation, reward, done, info = self.env.step(action_dict)

        obs = np.zeros((UNIQUE_OBJ_CNT,))
        for obj, state in enumerate(observation["context"]):
            if state:
                obs[self.obj2index[obj]] = 1
        obs[UNIQUE_OBJ_CNT - 1] = observation["light_state"]

        if not done:
            self.history.append(obs)

        obs = np.pad(
            np.concatenate((self.blicket_num, *self.history)),
            (
                0,
                self.obs_shape[0]
                - OBJ_MAX_BLICKET
                - UNIQUE_OBJ_CNT * len(self.history),
            ),
        )

        for obj, prob in enumerate(observation["last_belief"]):
            if obj in self.obj2index:
                obs[
                    OBJ_MAX_BLICKET + UNIQUE_OBJ_CNT *
                    MAX_TRIAL + self.obj2index[obj]
                ] = prob
        # if reward == 20:
        #     self.render()
        # if done:
        #     with open('file.txt', 'w+') as f:
        #         print(self.stepcntdict, file=f)
        return obs, reward, done, info

    def reset(self):
        self.blicket_num = np.zeros(OBJ_MAX_BLICKET)
        self.blicket_num[int(np.sum(self.env.true_label)) - 1] = 1.0
        observation = self.env.reset()
        count = 0
        self.history = []
        self.obj2index = {}
        self.index2obj = {}
        for index, avail in enumerate(observation["available_objects"]):
            if avail:
                self.obj2index[index] = count
                self.index2obj[count] = index
                count += 1

        obs = np.zeros((UNIQUE_OBJ_CNT,))
        for obj, state in enumerate(observation["context"]):
            if state:
                obs[self.obj2index[obj]] = 1
        obs[UNIQUE_OBJ_CNT - 1] = observation["light_state"]
        self.history.append(obs)

        obs = np.pad(
            np.concatenate((self.blicket_num, *self.history)),
            (
                0,
                self.obs_shape[0]
                - OBJ_MAX_BLICKET
                - UNIQUE_OBJ_CNT * len(self.history),
            ),
        )
        for obj, prob in enumerate(observation["last_belief"]):
            if obj in self.obj2index:
                obs[
                    OBJ_MAX_BLICKET + UNIQUE_OBJ_CNT *
                    MAX_TRIAL + self.obj2index[obj]
                ] = prob

        return obs

    def render(self):
        pass


class VisualIVRE(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        bpy.ops.wm.open_mainfile(
            filepath=str(Path(__file__).parent /
                         "render/data/blicket_scene.blend")
        )
        self.env = env

        self.action_shape = (2 * (UNIQUE_OBJ_CNT - 1),)

        self.obs_shape = (
            IMG_HEIGHT * IMG_WIDTH * ((UNIQUE_OBJ_CNT - 1) + MAX_TRIAL) * 3
            + (UNIQUE_OBJ_CNT - 1)
            + OBJ_MAX_BLICKET,
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.action_shape, dtype=np.float32
        )

        self.img_base = np.array(
            Image.open(Path(__file__).parent / "render/data/empty_scene.png").convert(
                "RGB"
            )
        )

        self.action = None
        self._render_init()

    def _render_init(self):
        # Load materials
        bpy.context.preferences.edit.undo_memory_limit = 10
        bpy.context.preferences.edit.use_global_undo = False
        bpy.context.preferences.edit.undo_steps = 1

        utils.load_materials(Path(__file__).parent / "render/data/materials")

        # Set render arguments so we can get pixel coordinates later.
        # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
        # cannot be used.
        render_args = bpy.context.scene.render
        render_args.engine = "BLENDER_EEVEE"

        render_args.resolution_x = IMG_WIDTH
        render_args.resolution_y = IMG_HEIGHT
        render_args.resolution_percentage = 100

        bpy.context.scene.camera = bpy.data.objects["Camera"]

        # Put a plane on the ground so we can compute cardinal directions
        bpy.ops.mesh.primitive_plane_add(size=5, calc_uvs=False)
        plane = bpy.context.object
        # plane.hide_viewport = True
        # plane.hide_render = True

        # Figure out the left, up, and behind directions along the plane and record
        # them in the scene structure
        camera = bpy.data.objects["Camera"]
        plane_normal = plane.data.vertices[0].normal

        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))

        utils.delete_object(plane)

        plane_behind = (
            cam_behind - cam_behind.project(plane_normal)).normalized()
        plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
        plane_up = cam_up.project(plane_normal).normalized()

        scene_struct = {
            "camera": {},
            "light_state": None,
            "objects": [],
            "directions": {},
        }

        # Save camera info in case it is needed
        scene_struct["camera"]["location"] = tuple(camera.location)
        scene_struct["camera"]["rotation"] = tuple(camera.rotation_euler)

        # Save all six axis-aligned directions in the scene struct
        scene_struct["directions"]["behind"] = tuple(plane_behind)
        scene_struct["directions"]["front"] = tuple(-plane_behind)
        scene_struct["directions"]["left"] = tuple(plane_left)
        scene_struct["directions"]["right"] = tuple(-plane_left)
        scene_struct["directions"]["above"] = tuple(plane_up)
        scene_struct["directions"]["below"] = tuple(-plane_up)

        scene_struct["key_light"] = [
            bpy.data.objects["Lamp_Key"].location[i] for i in range(3)
        ]
        scene_struct["back_light"] = [
            bpy.data.objects["Lamp_Back"].location[i] for i in range(3)
        ]
        scene_struct["fill_light"] = [
            bpy.data.objects["Lamp_Fill"].location[i] for i in range(3)
        ]

        self.scene_struct = scene_struct

        cnt = 0
        for mat in bpy.data.materials:
            bpy.data.objects["Unused"].material_slots[cnt].material = mat
            cnt += 1
        # switch on nodes
        bpy.context.scene.use_nodes = True

        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new("CompositorNodeRLayers")
        rl.location = 0, 0

        # create output node
        v = tree.nodes.new("CompositorNodeViewer")
        v.location = 0, 0
        v.use_alpha = False

        # Links
        # link Image output to Viewer input
        links.new(rl.outputs[0], v.inputs[0])
        links.new(rl.outputs[1], v.inputs[1])

        warmup()

    def _img_postprocess(self, image, gamma=2.2):
        image = image.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[::-1, :, :-1]
        image = np.clip((((image) ** (1 / gamma)) * 255.0), 0, 255.0)
        return image.astype(np.uint8)

    def _render(self, ctx):
        return None

    def _clean(self):
        """
        Explict garbage collection.
        """
        utils.removeDataBlocks()
        # bpy.ops.outliner.orphans_purge(do_recursive=True)

    def reset(self):
        self._clean()
        observation = self.env.reset()
        count = 0
        self.history = []
        self.obj2index = {}
        self.index2obj = {}
        for index, avail in enumerate(observation["available_objects"]):
            if avail:
                self.obj2index[index] = count
                self.index2obj[count] = index
                count += 1
        self.history = np.empty(
            [MAX_TRIAL, IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        self.objects = np.empty(
            [UNIQUE_OBJ_CNT - 1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8
        )
        self.blicket_num = np.zeros(OBJ_MAX_BLICKET)
        self.blicket_num[int(np.sum(self.env.true_label)) - 1] = 1.0

        for i in range(MAX_TRIAL):
            self.history[i] = self.img_base
        total_objs = 0
        for i, flag in enumerate(self.env.available_objects):
            if flag:
                img = Image.open(
                    Path(__file__).parent / f"render/data/objects/all_{i}.png"
                ).convert("RGB")
                img = np.array(img.resize((160, 120)))
                self.objects[total_objs] = img
                total_objs += 1
        self.render()
        obs = {
            "objects": self.objects,
            "context": self.history,
            "last_belief": self.env.lastBelief,
            "blicket_num": self.blicket_num,
        }
        return self.flatten(obs)

    def flatten(self, obs):
        objects = obs["objects"].flatten()
        context = obs["context"].flatten()
        last_belief = np.empty((UNIQUE_OBJ_CNT - 1,))
        for obj, prob in enumerate(obs["last_belief"]):
            if obj in self.obj2index:
                last_belief[self.obj2index[obj]] = prob
        blicket_num = obs["blicket_num"]

        return np.concatenate(
            (objects, context, last_belief, blicket_num), dtype=np.float32
        )

    def step(self, action):
        action = action / 2.0 + 0.5
        action = np.clip(action, 0.0, 1.0)
        belief = np.zeros(self.env.config_size, dtype=np.float32)
        trial = np.zeros(self.env.config_size, dtype=np.float32)
        for index, prob in enumerate(action[: UNIQUE_OBJ_CNT - 1]):
            belief[self.index2obj[index]] = prob

        for index, prob in enumerate(action[UNIQUE_OBJ_CNT - 1:]):
            trial[self.index2obj[index]] = prob > 0.5

        action_dict = {"belief": belief, "trial": trial}

        # observation, reward, done, info = self.env.step(action_dict)

        self.action = action_dict
        _, reward, done, info = self.env.step(self.action)

        if not done:
            self.render()

        obs = {
            "objects": self.objects,
            "context": self.history,
            "last_belief": self.env.lastBelief,
            "blicket_num": self.blicket_num,
        }

        return self.flatten(obs), reward, done, info

    def render(self):
        panel = self.env.history[self.env.stepcnt]

        added_objects = generate_scene(
            panel.light_state, panel.objects, self.scene_struct
        )

        bpy.ops.render.render(write_still=False)

        image = np.array(bpy.data.images["Viewer Node"].pixels)

        image = self._img_postprocess(image)

        img = np.array(
            image,
            dtype=np.float32,
        )
        self.history[self.env.stepcnt] = img


# if __name__ == "__main__":
#     from PIL import Image
#     """
#     Test
#     """
#     env = SymbolicIVRE(IVRE())
#     import time
#     history_time = []

#     with open('hist.txt', 'w+') as f:
#         for epi in range(1):
#             t1 = time.time()
#             obs = env.reset()
#             for step in range(1):
#                 # print("-" * 100)

#                 # print(f"Step {step}:")
#                 # print("----------------actor---------------")
#                 # print("Step {}".format(step))
#                 obs, reward, done, info = env.step(env.action_space.sample())
#                 print(f"{len(obs)} len")
#                 # print("obs=", obs, "reward=", reward.shape, "done=", done)
#                 if done:
#                     print("Game Over!", "reward=", reward)

#                     break
#                 # print("\n\n")
#             time_usage = time.time() - t1
#             print(f"time : {time_usage}")
#             history_time.append(time_usage)
#             # f.write(str(time_usage)+'\n')
#     print(history_time)
#     # from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    from PIL import Image

    """
    Test
    """
    env = VisualIVRE(IVRE())
    # print(f"shape is {obs[0].shape}")``
    # for cnt, img in enumerate(obs[0]):
    #     Image.fromarray(obs[0][cnt].astype(np.uint8)).save(
    # f'{Path(__file__).parent}/render/rendered_images/output_{cnt}.png')
    import time

    history_time = []
    from einops import rearrange

    with open("hist.txt", "w+") as f:
        for epi in range(1):
            t1 = time.time()
            obs = env.reset()
            for step in range(10000):
                # print("-" * 100)

                # print(f"Step {step}:")
                # print("----------------actor---------------")
                # print("Step {}".format(step))
                obs, reward, done, info = env.step(env.action_space.sample())
                obs = rearrange(
                    obs[: 120 * 160 * 3 * 19],
                    "(p c h w) -> (p) h w c",
                    c=3,
                    h=IMG_HEIGHT,
                    w=IMG_WIDTH,
                    p=19,
                )
                print(obs.shape)
                Image.fromarray(obs[step].astype(np.uint8)).save(
                    f"{Path(__file__).parent}/render/rendered_images/output_step_{os.getpid()}_{epi}_{step}.png"
                )
                # print("obs=", obs, "reward=", reward.shape, "done=", done)
                if done:
                    print("Game Over!", "reward=", reward)

                    break
                # print("\n\n")
            time_usage = time.time() - t1
            print(f"time : {time_usage}")
            history_time.append(time_usage)
            # f.write(str(time_usage)+'\n')
    print(history_time)
    # from stable_baselines3.common.env_checker import check_env
