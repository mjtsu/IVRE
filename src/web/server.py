from flask import Flask, request, render_template, redirect, url_for
from IVRECoreWeb import IVRE
from random import randint
import numpy as np
from PIL import Image
from blicket import BlicketView
from flask import g
import os
from wtforms import SubmitField
import sys
from pathlib import Path
import json
from flask_wtf import FlaskForm
sys.path.append(str(Path(__file__).parent/'../'))
# coding=utf-8

user_list = []
# GLOBAL
flag = "belief"
num = -1
reward = 0
total_reward = 0
choices_48 = []
user_id = 0
finished = 0

app = Flask(__name__)
app.config['SECRET_KEY'] = '382a6037-5327-4a04-9d73-6d4cb0337297'


def ivre_init():
    global ivre
    global num
    global user_id
    global reward
    global total_reward
    global finished
    ivre = IVRE()
    print("init!!!")
    reward = 0
    total_reward = 0
    num = 0
    user_id = randint(1, 10000000)
    app.logger.info('User %d//Init//Objs:%s', user_id,
                    str(ivre.blicketProblem.objs))
    app.logger.info('User %d//Init//True_label:%s', user_id,
                    str(np.concatenate((np.argwhere(ivre.true_label == 1.))).tolist()))
    while num < 4:
        choices = ivre.history[num].objects
        state = 'off'
        for obj in choices:
            if ivre.true_label[obj]:
                state = 'on'
        app.logger.info('User %d//Epoch:%d//Obs:%s',
                        user_id, num, str(choices))
        # render_fig(choices, num, state)
        num += 1
    ivre.stepcnt = 4
    finished = 0


@app.route('/', methods=["GET", "POST"])
# introductin to est
def hello():
    ivre_init()
    return redirect(url_for('intro'))


@app.route('/game', methods=["GET", "POST"])
# introductin to ivre
def intro():
    global flag
    global num
    global reward
    global total_reward
    global ivre
    global belief
    global finished

    class PostForm(FlaskForm):
        guess = SubmitField('Guess!')
        trial = SubmitField('Trial!')
    form = PostForm()
    if num == -1:
        ivre_init()
    if form.guess.data:
        choices = request.form.getlist('bt')
        choices_10h = []
        choices_48 = []
        for obj in choices:
            choices_10h.append(int(obj))
            choices_48.append(ivre.blicketProblem.objs[int(obj)])
        if set(choices_48) == set(np.concatenate(np.argwhere(ivre.true_label == 1.)).tolist()):
            reward = 1
        else:
            reward = 0
        belief = choices_48
        reward = float("%.2f" % round(ivre.reward(belief), 2))
        total_reward += reward
        app.logger.info('User %d//Epoch:%d//Belief:%s//Reward:%d',
                        user_id, num, str(choices_48), reward)
        if num == 10 or reward == 1:
            # print("End, Your reward is: ")
            finished = 1
            app.logger.info('User %d//Epoch:%d//GameEnd', user_id, num)
            # return redirect(url_for('new_game'))
        else:
            flag = "trial"
    if form.trial.data:
        choices = request.form.getlist('bt')
        choices_10h = []
        choices_48 = []
        for obj in choices:
            choices_10h.append(int(obj))
            choices_48.append(ivre.blicketProblem.objs[int(obj)])
        state = 'off'
        for obj in choices_48:
            if ivre.true_label[obj]:
                state = 'on'
        app.logger.info('User %d//Epoch:%d//Trial:%s//State:%s',
                        user_id, num, str(choices_48), state)
        ivre.step({"trial": choices_48, "belief": belief})
        flag = "belief"
        num += 1
    kwargs = {
        "flag": flag,
        "num": num,
        "reward": reward,
        "total_reward": total_reward,
        "eps_id": ivre.eps_id,
        "form": form,
        "finished": finished,
    }
    return render_template('new_page.html', **kwargs)


@app.route("/pic/<string:cate>/<string:id>/<string:seq>")
def get_pic_path(cate, id, seq):
    prefix = "../static/eps"
    if cate == "base":
        if int(seq) != 101 and int(seq) != 102:
            f = open(f"static/eps/configs/{id}/info.json",)
            info_json = json.load(f)[0]
            objs = info_json["all_objs"]
            return f"{prefix}/{cate}/{id}/base_{objs[int(seq)]}_no.png"
        elif int(seq) == 101:
            files = os.listdir(f"static/eps/base/{id}")
            for i in range(len(files)):
                if 'blickets_' in files[i]:
                    return f"{prefix}/{cate}/{id}/{files[i]}"
        elif int(seq) == 102:
            files = os.listdir(f"static/eps/base/{id}")
            for i in range(len(files)):
                if 'nonblickets_' in files[i]:
                    return f"{prefix}/{cate}/{id}/{files[i]}"
    elif cate == "figs":
        base_list = os.listdir(f"static/eps/figs/{id}")
        current_objs = sorted(ivre.history[int(seq)].objects)
        if len(current_objs) == 0:
            return "../static/none.png"
        name_prefix = "_".join([str(i) for i in current_objs])
        if name_prefix + "_on.png" in base_list:
            return f"{prefix}/{cate}/{id}/{name_prefix}_on.png"
        else:
            return f"{prefix}/{cate}/{id}/{name_prefix}_off.png"


@app.route("/new/", methods=["GET", "POST"])
def new_game():
    global flag
    global num
    global reward
    global total_reward
    num = -1
    flag = "belief"
    reward = 0
    total_reward = 0
    return redirect(url_for('intro'))


if __name__ == "__main__":
    app.config["JSON_AS_ASCII"] = False
    app.run(host="127.0.0.1", port=8080, debug=True)
