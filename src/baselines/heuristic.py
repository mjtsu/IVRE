import numpy as np
import matplotlib.pyplot as plt
from src.const import (
    OBJ_MAX_BLICKET,
    UNIQUE_OBJ_CNT,
    MAX_TRIAL, EPS,
    OBJ_PANEL_MAX
)
from src.IVRECore import IVRE
from src.IVRE import SymbolicIVRE
import random
from tqdm import trange


class heuristicSolver:
    def __init__(self, obs) -> None:
        self.history = []
        for i in range(MAX_TRIAL):
            self.history.append(
                obs[
                    OBJ_PANEL_MAX
                    + i * UNIQUE_OBJ_CNT: OBJ_PANEL_MAX
                    + (i + 1) * UNIQUE_OBJ_CNT
                ]
            )

    def _check(self, solution):
        for view in self.rules:
            predicted = np.sum(solution * view[:-1]) > 0
            if predicted != view[-1]:
                return False

        return True

    def _solver(self, obj_index):
        if obj_index == UNIQUE_OBJ_CNT - 1:
            if self._check(self.solution):
                self.solcnt += 1
                self.problogits += self.solution

            return True

        for i in range(2):
            self.solution[obj_index] = i
            self._solver(obj_index + 1)

        return self.problogits

    def OracleSolver(self):
        self.problogits = np.zeros((UNIQUE_OBJ_CNT - 1,), dtype=np.int32)
        self.solution = np.zeros((UNIQUE_OBJ_CNT - 1,), dtype=np.int32)
        self.solcnt = 0
        self.rules = self.history
        result = self._solver(0)
        # print(result)
        # print(self.solcnt)
        return result / (self.solcnt + EPS)

    def action(self):
        belief = self.OracleSolver()
        confidence = np.abs(belief - 0.5)
        trial = np.zeros((UNIQUE_OBJ_CNT - 1,))
        trial[np.argsort(confidence)[:1]] = 1
        # for random trial:
        # trial  = np.random.normal(loc=0.5, scale=.1, size=(UNIQUE_OBJ_CNT-1, ))
        action = np.concatenate((belief, trial))
        action = action * 2.0 - 1.0
        return action


if __name__ == "__main__":
    env = SymbolicIVRE(IVRE())
    reward_sum = 0
    sol_cnt = 0
    steps_solved = []
    for eps in trange(10000):
        reward = 0
        obs = env.reset()
        action = heuristicSolver(obs).action()
        for i in range(MAX_TRIAL):
            obs, reward_step, done, _ = env.step(action)
            if reward_step > EPS:
                sol_cnt += 1
            reward += reward_step
            if done:
                if reward_step == 20:
                    steps_solved.append(i)
                    env.render()
                break
            action = heuristicSolver(obs).action()
        reward_sum += reward
    print(
        len(steps_solved),
        dict((l, steps_solved.count(l) / len(steps_solved))
             for l in set(steps_solved)),
    )
    print(
        f"Episode {eps} : {reward}, Avg {reward_sum / (eps + 1)}, Psr{sol_cnt / (eps + 1)}"
    )
