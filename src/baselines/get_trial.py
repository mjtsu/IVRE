import random
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from src.const import (
    ALL_CONFIG_SIZE,
    OBJ_MIN_BLICKET,
    OBJ_MAX_BLICKET,
    OBJ_PANEL_MIN,
    OBJ_PANEL_MAX,
    UNIQUE_OBJ_CNT,
    CONTEXT_PER_SEQ,
    MAX_TRIAL,
)
import heapq
import src.baselines.opt_models as opt_models


# def human_trial_input(machine):
#     objs = [int(n) for n in input("Enter your trial: ").split(" ")]
#     return objs


def human_trial_input(env):
    print(env.action_space)
    objs = [int(n) for n in input("Enter your trial: ").split(" ")]
    belief = (
        env.lastBelief
        if (env.lastBelief)
        else env.action_space.sample()[: env.config_size]
    )
    trial = [0] * env.config_size
    for i in objs:
        trial[i] = 1
    return belief + trial


# def random_trial_input(machine):
#     objs = random.sample(machine.objs, random.randint(1, UNIQUE_OBJ_CNT))
#     return objs


def random_trial_input(env):
    # lis = [i for i in range(9)]
    # num = int(np.sum(env.env.true_label))
    # belief_idx = random.sample(lis, num)
    # belief = np.zeros((UNIQUE_OBJ_CNT-1, )) - 1
    # belief[belief_idx] = 1
    # trial_idx = random.sample(lis, num)
    # trial = np.zeros((UNIQUE_OBJ_CNT-1, )) - 1
    # trial[trial_idx] = 1
    # return np.concatenate((belief, trial))
    belief = np.random.normal(loc=0, scale=1, size=(UNIQUE_OBJ_CNT - 1,))
    trial = np.random.normal(loc=0, scale=1, size=(UNIQUE_OBJ_CNT - 1,))

    return np.concatenate((belief, trial))


def bayes_trial_input(env):
    trial = [0] * (UNIQUE_OBJ_CNT - 1)
    belief = [-1] * (UNIQUE_OBJ_CNT - 1)
    his_x = []
    his_y = []
    tmp = [0.0] * 9
    his_x.append(tmp)
    his_y.append(0.0)
    tmp = [1.0] * 9
    his_x.append(tmp)
    his_y.append(1.0)
    for i in range(env.env.stepcnt):
        if not env.history[i][-1]:
            for j in range(10):
                tmp = env.history[i][:-1]
                his_x.append(list(tmp))
                his_y.append(env.history[i][-1])
        else:
            tmp = env.history[i][:-1]
            his_x.append(list(tmp))
            his_y.append(env.history[i][-1])
    bnb = BernoulliNB()
    bnb.fit(np.array(his_x), np.array(his_y))
    belief_proba = []
    for i in range(UNIQUE_OBJ_CNT - 1):
        tmp = [0.0] * (UNIQUE_OBJ_CNT - 1)
        tmp[i] = 1.0
        pre = bnb.predict_proba(np.array([tmp]))[0][1]
        belief_proba.append(pre)
    blicket_num = int(np.sum(env.env.true_label))
    max_number = heapq.nlargest(blicket_num, belief_proba)
    t = 0
    while t < blicket_num:
        for num in max_number:
            index = belief_proba.index(num)
            belief[index] = 1
            t += 1
    step = env.stepcnt
    trial[step - 1] = 1
    return np.array(list(belief) + list(trial))


def opt_trial_input(env):
    trial = [-1] * (UNIQUE_OBJ_CNT - 1)
    belief = [-1] * (UNIQUE_OBJ_CNT - 1)
    NoTearsLinear = opt_models.NoTearsLinear(0.5, 0.5)
    NoTearsLinear.train(np.array(env.history))
    t = 0
    blicket_num = int(np.sum(env.env.true_label))
    pre = NoTearsLinear.test([i for i in range(UNIQUE_OBJ_CNT - 1)])
    for i, res in enumerate(pre):
        if res:
            while t < blicket_num:
                belief[i] = res
                t += 1
    step = env.stepcnt
    trial[step - 1] = 1
    return np.array(list(belief) + list(trial))


def lazy_trial_input(env):
    trial = [0.5] * (UNIQUE_OBJ_CNT - 1)
    belief = [0.5] * (UNIQUE_OBJ_CNT - 1)
    for i in range((UNIQUE_OBJ_CNT - 1)):
        trial[i] = random.randint(40, 60) / 100
        belief[i] = random.randint(40, 60) / 100
    last_belief = []
    if env.env.stepcnt:
        last_belief = env.env.lastBelief
    for obj, prob in enumerate(last_belief):
        if env.env.available_objects[obj]:
            belief[env.obj2index[obj]] = prob
    # print(belief)
    state = env.history[env.env.stepcnt][-1]
    trial = [-1] * (UNIQUE_OBJ_CNT - 1)
    for i in range(len(env.history[env.env.stepcnt]) - 1):
        if state == 1 and 0 < belief[i] < 0.99 and env.history[env.env.stepcnt][i] == 1:
            if env.history[env.env.stepcnt][:-1].sum() == 1:
                belief[i] = 1
            else:
                belief[i] = min(belief[i] + 0.2, 0.98)
        elif state == 0 and env.history[env.env.stepcnt][i] == 1:
            if env.history[env.env.stepcnt][:-1].sum() == 1:
                belief[i] = 0
            else:
                belief[i] = max(belief[i] - 0.2, 0)
    if env.env.stepcnt >= 3:
        for j in range((UNIQUE_OBJ_CNT - 1)):
            if 0.2 < belief[j] < 0.99:
                trial[j] = 1
                break
    return np.array(list((np.array(belief) - 0.5) * 2) + list(trial))
