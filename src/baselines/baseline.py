from src.IVRECore import IVRE
from src.IVRE import SymbolicIVRE
import src.baselines.get_trial as get_trial
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import trange


def main(args):
    marks = []
    trials = []
    sol_cnt = 0
    for i in trange(args.episode):
        total_reward = 0
        env = SymbolicIVRE(IVRE())
        obs = env.reset()
        # env.render()
        for step in range(args.trial_time):
            # print("Step {}".format(step + 1))
            action = getattr(get_trial, args.trial_model)(env)
            obs, reward, done, info = env.step(action)
            # print('obs=', obs, 'reward=', reward, 'done=', done)
            # env.render()
            total_reward += reward
            if done:
                # print("Game Over!", "reward=", total_reward)
                if reward == 20:
                    # env.render()
                    sol_cnt += 1
                break
        marks.append(total_reward)
        trials.append(step)
    plt.plot([x for x in range(len(marks))], marks)
    plt.plot([x for x in range(len(marks))], trials)
    # plt.show()
    print("Average mark: ", np.mean(marks), "Sol_cnt: ", sol_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial_time", default=10, type=int, help="The trial num in one episode."
    )
    parser.add_argument(
        "--trial_model",
        type=str,
        default="opt_trial_input",
        help="the model to get the trial",
    )
    parser.add_argument(
        "--visualize", type=str, default="False", help="gen the picture or not"
    )
    parser.add_argument("--episode", type=int, default=1,
                        help="episode number")
    args = parser.parse_args()
    main(args)
