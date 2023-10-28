# <img src="./figs/mammal.svg" height="35"/> IVRE: <u>__I__</u>nteractive <u>__V__</u>isual <u>__RE__</u>easoning under Uncertainty
Manjie Xu*, Guangyuan Jiang*, Wei Liang, Chi Zhang, Yixin Zhu 
<a href='https://arxiv.org/abs/2206.09203'>
  <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
</a>
</a>
<a href='https://sites.google.com/view/ivre/home'>
  <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
</a>
<a href='https://youtu.be/h3FFcWkTHxI'>
  <img src='https://img.shields.io/badge/Project-Demo-red?style=plastic&logo=Youtube&logoColor=red' alt='Project Page'>
</a>

<div align=center>
<img src=./figs/intro.png width=400/>
</div>

This is the offical implementation of our paper IVRE in NeurIPS 2023 D&B Track. 

# <img src="./figs/mammal.svg" height="35"/>IVRE Introduction

One of the fundamental cognitive abilities of humans is to quickly resolve uncertainty by generating hypotheses and testing them via active trials. Encountering a novel phenomenon accompanied by ambiguous cause-effect relationships, humans make hypotheses against data, conduct inferences from observation, test their theory via experimentation, and correct the proposition if inconsistency arises. 

These iterative processes persist until the underlying mechanism becomes clear. In this work, we devise the <img src="./figs/mammal.svg" height="15"/>**IVRE** (pronounced as "Ivory") environment for evaluating artificial agents' reasoning ability under uncertainty. <img src="./figs/mammal.svg" height="15"/>**IVRE** is an interactive environment featuring rich scenarios centered around *Blicket* detection. Agents in <img src="./figs/mammal.svg" height="15"/>**IVRE** are placed into environments with various ambiguous action-effect pairs and asked to figure out each object's role. Agents are encouraged to propose effective and efficient experiments to validate their hypotheses based on observations and gather more information. The game ends when all uncertainties are resolved or the maximum number of trials is consumed. 

By evaluating modern artificial agents in <img src="./figs/mammal.svg" height="15"/>**IVRE**, we notice a clear failure of today's learning methods compared to humans. Such inefficacy in interactive reasoning ability under uncertainty calls for future research in building humanlike intelligence.

# <img src="./figs/mammal.svg" height="35"/>IVRE benchmark
An example of IVRE benchmark is provided below.
<img src=./figs/env.png>
In each episode of IVRE, an agent is presented with novel observations and asked to figure out all objects' Blicketness. The agent will firstly be shown with some observations(the so-called 'Context'). After that, the agent proposes new experiments(the 'Trial') to validate its hypothesis and updates its current belief. 

# <img src="./figs/mammal.svg" height="35"/>IVRE TODO ![coverage](https://img.shields.io/badge/coverage-80%25-yellowgreen) ![version](https://img.shields.io/badge/version-1.0.0-purple)

- [x] <img src="./figs/mammal.svg" height="15"/>IVRE environment code.
- [x] <img src="./figs/mammal.svg" height="15"/>IVRE baselines (symbolic & visual).
- [x] <img src="./figs/mammal.svg" height="15"/>IVRE bpy render code.
- [x] <img src="./figs/mammal.svg" height="15"/>IVRE web version.
- [ ] <img src="./figs/mammal.svg" height="15"/>IVRE checkpoints.

# Build <img src="./figs/mammal.svg" height="35"/>IVRE env
- Clone this repo.

  ```
  git clone
  ```
- Install dependencies. We recommend using conda.
  ```
  conda create -n ivre python=3.9
  conda activate ivre
  pip install -e .
  ```

For symbol-input agents, this should be enough.  
For image-input agents, you need to use Blender for rendering. We have compiled a Blender version for you. You can download it from [here](https://drive.google.com/file/d/12IL5tglxOg-RFbR-wP6_fO8j-atPa-4J/view). Unzip the file, rename it to `bpy` and put it under the `render` folder. Then you can run the following command to test if the rendering works.
```
# test bpy
cd src/render/bpy
python -c "import bpy; print(bpy.app.version_string)"
# 3.2.0
```
  
# <img src="./figs/mammal.svg" height="35"/>IVRE Baselines
### Heuristic Baselines
```
cd src/baselines
python baseline.py --trial_model {MODEL_NAME}
# MODEL_NAME: {human_trial_input, random_trial_input, bayes_trial_input, opt_trial_input, lazy_trial_input}
```
### Reinforcement Learning Baselines
```
cd src/baselines/rl_baselines
python {MODEL_NAME}.py
# MODEL_NAME: {ddpg, ppo,td3,rnn_ddpg,rnn_td3}
```
# Host your own <img src="./figs/mammal.svg" height="35"/>IVRE
IVRE can be hosted on a local server and accessed via a web browser. To do so, you need to install [Flask](https://flask.palletsprojects.com/en/2.0.x/) and [Flask-SocketIO](https://flask-socketio.readthedocs.io/en/latest/). You also need to install Blender for rendering.
```
cd src/web
python server.py
```
This will host IVRE on a local server. You can access it via a web browser by visiting `http://localhost:8080/`. If you have successfully installed Blender for rendering, you should be able to conduct infinite episodes in the web version of IVRE. Alternatively, you can also run IVRE based on rendered episodes. Download the rendered episodes from the google drive and put them in `src/web/static/eps`.

# Citation
If you find the paper and/or the code helpful, please cite us.
```
@inproceedings{xu2023interactive,
  title={Interactive Visual Reasoning under Uncertainty},
  author={Xu, Manjie and Jiang, Guangyuan and Liang, Wei and Zhang, Chi and Zhu, Yixin},
  booktitle={NeurIPS},
  year={2023}
}
```
