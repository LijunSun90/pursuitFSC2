"""
Modified based on Openai spinningup test_policy.py.
"""
import numpy as np
import time
import joblib
import os
import os.path as osp
import torch
import argparse
import PIL
from datetime import datetime
from actor_critic.common.logx import EpochLogger
from pursuit_game import pursuit_vsos as pursuit


# Modify the data_path.
data_path = "./results/sos_Actor-Critic/sos_actor_critic_p_s3/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('f_path', type=str, nargs='*', default=data_path)

    parser.add_argument('--use_seed', '-s', default=True)

    parser.add_argument('--n_episodes', '-n', type=int, default=100)
    parser.add_argument('--x_size', '-xs', type=int, default=40)
    parser.add_argument('--y_size', '-ys', type=int, default=40)
    parser.add_argument('--n_pursuers', '-np', type=int, default=8)
    parser.add_argument('--n_targets', '-nt', type=int, default=50)
    parser.add_argument('--episode_len', '-l', type=int, default=500)

    parser.add_argument('--render', '-r', default=False)
    parser.add_argument('--save_frames', '-sf', default=False)

    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    return parser.parse_args()


def run_policy(env, get_action, arg_list, n_pursuers, max_ep_len=None, num_episodes=100, use_seed=True,
               render=True, save_frames=False):
    assert env is not None, "Environment not found!\n\n It looks like the environment" \
                            "wasn't saved, and we can't run the agent in it. :( Check" \
                            "out the readthedocs page on Experiment Outputs fro how to" \
                            "handle this situation."

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # logger_output_fname = 'progress_test_' + timestamp + '.txt'
    logger_output_fname = "_".join(['progress_test_parallel',
                                    str(arg_list.x_size), str(arg_list.y_size),
                                    str(arg_list.n_pursuers), str(arg_list.n_targets), '.txt'])
    logger = EpochLogger(output_dir=data_path, output_fname=logger_output_fname)

    # Random seed.
    seed_list = range(num_episodes)
    seed = seed_list[0]
    if use_seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)

    env.reset()
    ep_ret, ep_len, ep_collisions, ep_collisions_with_obstacles, i_episode = 0, 0, 0, 0, 0
    agents_ep_ret = [0 for _ in range(n_pursuers)]
    frame_list = []

    while i_episode < num_episodes:
        if render:
            if not save_frames:
                env.render()
                time.sleep(1e-3)
            else:
                frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))

        actions = []
        dones = []
        for agent_id in range(n_pursuers):
            o, r, d, _ = env.last_vsos(agent_id)
            o = np.reshape(o, -1)
            agents_ep_ret[agent_id] += r

            a = get_action(torch.as_tensor(o, dtype=torch.float32))
            # a = np.random.randint(5)

            actions.append(a)
            dones.append(d)

        for agent_id in range(n_pursuers):
            d, a = dones[agent_id], actions[agent_id]
            if d:
                env.step(None)
            else:
                env.step(a)

        ep_len += 1
        ep_collisions += env.env.env.env.n_collision_events_per_multiagent_step

        timeout = (ep_len == max_ep_len)
        terminal = env.env.env.env.is_terminal or timeout
        if terminal:
            ep_ret = np.mean(agents_ep_ret)
            capture_rate = sum(env.env.env.env.evaders_gone) / len(env.env.env.env.evaders_gone)
            ep_collisions_with_obstacles = env.env.env.env.n_collision_with_obstacles
            logger.store(EpRet=ep_ret, EpLen=ep_len, CaptureRate=capture_rate,
                         Collisions=ep_collisions, CollideObstacles=ep_collisions_with_obstacles)

            if use_seed:
                print('Seed %3d \t Episode %3d \t EpRet %.3f \t EpLen %d \t CaptureRate %.3f \t Collisions %d \t CollideObstacles %d' %
                      (seed, i_episode, ep_ret, ep_len, capture_rate, ep_collisions, ep_collisions_with_obstacles))
            else:
                print('Episode %3d \t EpRet %.3f \t EpLen %d \t CaptureRate %d \t Collisions %.3f \t CollideObstacles %d' %
                      (i_episode, ep_ret, ep_len, capture_rate, ep_collisions, ep_collisions_with_obstacles))

            i_episode += 1
            if use_seed and i_episode < num_episodes:
                seed = seed_list[i_episode]
                torch.manual_seed(seed)
                np.random.seed(seed)
                env.seed(seed)

            if save_frames:
                now = datetime.now()
                time_format = "%Y%m%d%H%M%S"
                timestamp = now.strftime(time_format)
                filename = data_path + "result_pursuit_sos_" + \
                    str("{:.3f}".format(ep_ret)) + "_" + \
                    str("{:.3f}".format(capture_rate)) + "_" + \
                    str("{:d}".format(ep_collisions)) + "_" + \
                    str("{:d}".format(ep_collisions_with_obstacles)) + "_" + \
                    timestamp + ".gif"
                images_to_gif(frame_list, filename)

            frame_list = []

            env.reset()
            ep_ret, ep_len, ep_collisions, ep_collisions_with_obstacles = 0, 0, 0, 0
            agents_ep_ret = [0 for _ in range(n_pursuers)]

    logger.log_tabular('UseSeed', use_seed)
    logger.log_tabular("NEpisodes", num_episodes)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', with_min_and_max=True)
    logger.log_tabular("CaptureRate", with_min_and_max=True)
    logger.log_tabular("Collisions", with_min_and_max=True)
    logger.log_tabular("CollideObstacles", with_min_and_max=True)
    logger.dump_tabular()


def images_to_gif(image_list, output_name):
    output_dir = osp.join(data_path, 'pngs')
    os.system("mkdir " + output_dir)
    for idx, im in enumerate(image_list):
        im.save(output_dir + "/target" + str(idx) + ".png")

    os.system("/usr/local/bin/ffmpeg -i " + output_dir + "/target%d.png " + output_name)
    os.system("rm -r " + output_dir)
    print('Write to file:', output_name)


def load_policy_and_env(f_path, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or Pytorch, along with RL env.
    Not exceptionally future-proof,  but it will suffice for basic use of the
    Spinning up implementations.
    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """
    # Handle which epoch to load from.
    if itr == 'last':
        # Check filenames for epoch (AKA iteration) numbers, find maximum value.
        pytsave_path = osp.join(f_path, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that is case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model'in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # Load the get_action function.
    get_action = load_pytorch_policy(f_path, itr, deterministic)

    # Try to load environment from save.
    # (Sometimes this will fail because the environment could not be picked.)
    try:
        state = joblib.load(osp.join(f_path, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(f_path, itr, deterministic=False):
    """Load a pytorch policy saved with Logger."""
    f_name = osp.join(f_path, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % f_name)
    model = torch.load(f_name)

    # Make function for producing an action given a single state.
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


if __name__ == '__main__':
    arg_list = parse_args()
    env, get_action = load_policy_and_env(arg_list.f_path,
                                          arg_list.itr if arg_list.itr >= 0 else 'last',
                                          arg_list.deterministic)

    # Instantiate environment.
    env = pursuit.env(max_cycles=arg_list.episode_len,
                      x_size=arg_list.x_size, y_size=arg_list.y_size,
                      n_evaders=arg_list.n_targets, n_pursuers=arg_list.n_pursuers,
                      obs_range=11,
                      surround=False, n_catch=1,
                      freeze_evaders=True)

    run_policy(env,
               get_action,
               arg_list,
               n_pursuers=arg_list.n_pursuers,
               max_ep_len=arg_list.episode_len,
               num_episodes=arg_list.n_episodes,
               use_seed=arg_list.use_seed,
               render=arg_list.render,
               save_frames=arg_list.save_frames)

    print("SUCCESS!")
