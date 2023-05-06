
import string
import random
import cv2
import numpy as np
from luxai_s2.env import LuxAI_S2
from agent import Agent

import pygame

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()


def animate(imgs, _return=True):
    video_name = ''.join(random.choice(string.ascii_letters) for i in range(18))+'.webm'
    print(video_name)
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width,height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    if _return:
        from IPython.display import Video
        return Video(video_name)

def interact(env, agents, steps):
   
    # reset our env
    obs = env.reset(seed=41)
    np.random.seed(0)
    imgs = []
    step = 0
    # Note that as the environment has two phases, we also keep track a value called 
    # `real_env_steps` in the environment state. The first phase ends once `real_env_steps` is 0 and used below

    # iterate until phase 1 ends
    done = False
    while env.state.real_env_steps < 0:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            if step==0:
                a = agents[player].bid_policy(step, o)
            else:
                a = agents[player].factory_placement_policy(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        #imgs += [env.render("rgb_array", width=640, height=640)]
        done = env.render("human", width=320, height=320)

    while not done:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        done = env.render("human", width=320, height=320)
        #img = env.render("rgb_array", width=640, height=640)
        #imgs += [img]
        done = done or dones["player_0"] and dones["player_1"]

    pygame.quit()
    
    return imgs

if __name__ == "__main__":
    
    env = LuxAI_S2() # create the environment object
    #env.py_visualizer.init_window()
    obs = env.reset(seed=0) # resets an environment with a seed
    agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
    imgs = interact(env, agents, 1000)
    #animate(imgs, _return=False)

