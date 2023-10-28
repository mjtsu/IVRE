# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

from src.const import (
    ALL_CONFIG_SIZE,
    OBJ_MIN_BLICKET,
    OBJ_MAX_BLICKET,
    UNIQUE_OBJ_CNT,
    CONTEXT_PER_SEQ,
    OBJ_PANEL_MIN,
    OBJ_PANEL_MAX,
)


class BlicketView(object):
    def __init__(self, light_state="no"):
        # light state: "no" for no light, "off" for light off, "on" for light on
        self.objects = []
        self.light_state = light_state

    def add_objects(self, objects):
        for obj in objects:
            self.objects.append(obj)

    def remove_objects(self, objects):
        for obj in objects:
            self.objects.remove(obj)

    def __repr__(self):
        return "BlicketView(objects={}, light_state={})".format(
            self.objects, self.light_state
        )


class BlicketQuestion(object):
    def __init__(
        self, min_potential_blickets, max_potential_blickets, config_size, shuffle
    ):
        # self.min_non_blickets = min_non_blickets
        # self.max_non_blickets = max_non_blickets
        self.config_size = config_size
        self.shuffle = shuffle

        blicket_num = random.randint(
            min_potential_blickets, max_potential_blickets)
        # non_blicket_num = random.randint(self.min_non_blickets, self.max_non_blickets)
        non_blicket_num = UNIQUE_OBJ_CNT - blicket_num - 1

        samples = random.sample(
            list(range(self.config_size)), k=UNIQUE_OBJ_CNT - 1)

        self.objs = samples
        self.blickets = samples[:blicket_num]
        self.non_blickets = samples[blicket_num:]
        self.contexts = []
        self.true_label = {}
        for obj in self.blickets:
            self.true_label[obj] = 1
        for obj in self.non_blickets:
            self.true_label[obj] = 0

    def gen_context(self):
        contexts = []
        for i in range(CONTEXT_PER_SEQ):
            obj_num = random.randint(OBJ_PANEL_MIN, OBJ_PANEL_MAX)
            context_sample = random.sample(self.objs, k=obj_num)
            flag = 0
            for obj in context_sample:
                if obj in self.blickets:
                    flag = 1
            if flag:
                view = BlicketView("on")
                view.add_objects(context_sample)
                contexts.append(view)
            else:
                view = BlicketView("off")
                view.add_objects(context_sample)
                contexts.append(view)
        self.contexts = contexts
        return contexts

    def get_views(self):
        views = self.gen_context()
        return views


def serialize(machines, n):
    question_list = []
    for machine in machines:
        view_list = []
        for i in range(n):
            json_dict = {}
            json_dict["light_state"] = machine.contexts[i].light_state
            json_dict["objects"] = machine.contexts[i].objects
            json_dict["all_objs"] = machine.objs
            view_list.append(json_dict)
        question_list.append(view_list)
    return question_list


def config_control(size, config_size):
    machines = []
    for _ in range(size):
        blicket_machine = BlicketQuestion(
            OBJ_MIN_BLICKET, OBJ_MAX_BLICKET, config_size, True
        )
        context_views = blicket_machine.get_views()
        # blicket_machine.sanity_check()
        # blicket_machine.check_blickets(context_views)
        machines.append(blicket_machine)
        return machines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_size", default=1, type=int, help="The training set size."
    )
    parser.add_argument(
        "--output_dataset_dir",
        default="../../ACRE_IID/config",
        type=str,
        help="The directory to save output dataset json files.",
    )
    parser.add_argument(
        "--seed", default=12345, type=int, help="The random number seed"
    )
    parser.add_argument(
        "--regime", default="IID", type=str, help="Regime could be IID, Comp, or Sys"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.isdir(args.output_dataset_dir):
        os.makedirs(args.output_dataset_dir)

    train_size = args.train_size

    questions = config_control(train_size, ALL_CONFIG_SIZE)
    with open(os.path.join(args.output_dataset_dir, "log.json"), "w") as f:
        json.dump(serialize(questions, CONTEXT_PER_SEQ), f, indent=4)
