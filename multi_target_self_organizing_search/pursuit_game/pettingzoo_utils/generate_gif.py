import random
import time
import numpy as np
from PIL import Image
import os
import sys
import subprocess
import shutil

def generate_gif(env, name, num_cycles=100, reset=False, fps=30):
    imgs = []
    env.reset()
    for step in range(num_cycles):
        for agent in env.agent_iter(env.num_agents):  # step through every agent once with observe=True
            obs, rew, done, info = env.last()
            if done:
                action = None
            elif isinstance(obs, dict) and 'action_mask' in obs:
                action = random.choice(np.flatnonzero(obs['action_mask']))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        if not env.agents:
            if not reset:
                break
            env.reset()

        ndarray = env.render(mode='rgb_array')
        im = Image.fromarray(ndarray)
        imgs.append(im)
            # im.save(f"{dir}{str(step).zfill(5)}.png")

    env.close()
        # render_gif_image(name, fps)
    hundredths = max(int(fps/100), 4)
    img, *imgs = imgs
    img.save(fp=f"gifs/{name}.gif", format='GIF', append_images=imgs,
         save_all=True, duration=hundredths, loop=0, include_color_table=True, optimize=True)
