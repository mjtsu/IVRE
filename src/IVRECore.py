import gymnasium as gym
from gymnasium import spaces
from src.blicket import config_control, BlicketView
import numpy as np
from src.const import ALL_CONFIG_SIZE, MAX_TRIAL, UNIQUE_OBJ_CNT, CONTEXT_PER_SEQ, CONFIDENCE_THRES, EPS, IMG_WIDTH, IMG_HEIGHT
from scipy.stats import entropy


class IVRE(gym.Env):
    """IVRE Environment Core, based on OpenAI Gym"""

    def __init__(self, config_size=ALL_CONFIG_SIZE):
        """Initialize

        Args:
            config_size ([int], optional): Maximum unique looking objects. Defaults to ALL_CONFIG_SIZE: 48.
        """
        super(IVRE, self).__init__()

        self.config_size = config_size
        self.action_space = spaces.Dict(
            {
                "trial": spaces.MultiBinary(config_size),  # Agent's trial
                "belief": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config_size,),
                    dtype=np.float32,  # Agent's current belief
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "available_objects": spaces.MultiBinary(
                    config_size
                ),  # Available objects in current episode
                "context": spaces.MultiBinary(
                    config_size
                ),  # Last Trial/ Initial panel objects
                "light_state": spaces.MultiBinary(
                    1
                ),  # Last Trial/ Initial panel Blicket Machine State
                "last_belief": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config_size,),
                    dtype=np.float32,  # Agent's previous belief
                ),
            }
        )
        self.reset()

    @staticmethod
    def _jsd(p, q):
        """A naive implementation of Jensen-Shannon Distance

        Args:
            p ([np.array]): p
            q ([np.array]): q

        Returns:
            [np.float]: Jensen-Shannon Distance of p,q
        """
        p = np.asarray(p) + EPS
        q = np.asarray(q) + EPS
        p /= p.sum()
        q /= q.sum()
        m = (p + q) / 2
        divergence = np.clip((entropy(p, m) + entropy(q, m)) / 2, 0.0, 1.0)
        return np.sqrt(divergence)

    def _dict2array(self, obj_dict):
        """Helper function for converting object ids to binary vectors

        Args:
            obj_dict ([dict]): [dict of object ids]

        Returns:
            [np.array]: [binary vector]
        """
        arr = np.zeros(shape=(self.config_size,), dtype=np.float32)
        for obj, state in obj_dict.items():
            arr[obj] = state

        return arr

    def _list2array(self, obj_list):
        arr = np.zeros(shape=(self.config_size,), dtype=np.uint8)
        arr[obj_list] = 1

        return arr

    def _check(self, solution):
        """Calculate Blicket Machine State

        Args:
            solution ([type]): [Current assignment]

        Returns:
            [bool]: [Whether current assignment is consistent with observation]
        """
        for view in self.rules:
            predictLight = 0
            for objs in view.objects:
                predictLight += solution[objs]
            if bool(predictLight) != (view.light_state == "on"):
                return False

        return True

    def _solver(self, obj_index):
        """Iterative Search

        Args:
            obj_index ([int]):

        Returns:
            [list]: [Possible assignments that satisfie current constraints]
        """
        if obj_index == len(self.blicketProblem.objs):
            if self._check(self.prob):
                self.solcnt += 1
                for key, value in self.prob.items():
                    if key in self.solution:
                        self.solution[key] += value
                    else:
                        self.solution[key] = value
            return True

        for i in range(2):
            self.prob[self.blicketProblem.objs[obj_index]] = i
            self._solver(obj_index + 1)

        return self.solution

    def _OracleSolver(self):
        """Search-based Oracle Solver, calculates each object's blicketness probability.

        Returns:
            [list]: [predicted object's Blicketness for object in available_objects]
        """
        self.solution = {}
        self.prob = {}
        self.solcnt = 0
        self.rules = self.history[: self.stepcnt]
        result = self._solver(0)
        for key, value in result.items():
            result[key] = value / self.solcnt

        return result

    def _genTrial(self, trial):
        """Generate new panel and its result.

        Args:
            trial ([list]): [objects in the new panel]

        Returns:
            [BlicketView]: [new panel]
        """
        light = False
        panel = BlicketView("off")
        panel.add_objects(
            [
                index
                for index, val in enumerate(trial)
                if val and self.available_objects[index]
            ]
        )

        for obj in panel.objects:
            if self.true_label[obj]:
                panel.light_state = "on"
                break

        self.history.append(panel)

        return panel

    def _reward(self, belief):
        """Calculate Reward

        Args:
            belief ([dict]): [Agent's belief]

        Returns:
            [reward]: [reward at current step]
        """
        oracle = self._dict2array(self._OracleSolver())
        for index, prob in enumerate(belief):
            if (
                self.available_objects[index]
                and abs(self.true_label[index] - prob) >= CONFIDENCE_THRES
            ):
                return (
                    -self._jsd(oracle, belief) - 1.0
                )  # JSD of Agent's Blief and Oracle Solver's result

        return 20  # If Correct

    def step(self, action):
        """Apply an action

        Args:
            action ([dict]): [Agent's action]
        """
        trial, belief = action["trial"], action["belief"]
        belief = self.available_objects * belief
        trial = self.available_objects * trial
        self.lastTrial, self.lastBelief = trial, belief
        self.stepcnt += 1

        if self.stepcnt >= CONTEXT_PER_SEQ:
            self._genTrial(trial)

        observation = {
            "available_objects": self.available_objects,
            "context": self._list2array(self.history[self.stepcnt].objects),
            "light_state": np.array(
                [1 if self.history[self.stepcnt].light_state == "on" else 0]
            ),
            "last_belief": self.lastBelief,
        }
        reward = self._reward(belief)

        done = bool(self.stepcnt == MAX_TRIAL or reward > EPS)
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.blicketProblem = config_control(1, self.config_size)[0]
        self.true_label = self._dict2array(self.blicketProblem.true_label)
        self.history = self.blicketProblem.contexts
        self.solution = {}
        self.prob = {}
        self.stepcnt = 0
        self.solcnt = 0
        self.lastTrial = None
        self.available_objects = self._list2array(self.blicketProblem.objs)
        self.lastBelief = np.zeros((self.config_size,), dtype=np.float32) + 0.5

        observation = {
            "available_objects": self.available_objects,
            "context": self._list2array(self.history[self.stepcnt].objects),
            "light_state": np.array(
                [1 if self.history[self.stepcnt].light_state == "on" else 0]
            ),
            "last_belief": self.lastBelief,
        }

        return observation
