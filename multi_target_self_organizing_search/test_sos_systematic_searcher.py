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


data_path = "./results/sos_systematic_searcher/"


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


def plan_path_to_corner(start_point, scope_1st_d, scope_2nd_d):
    # Oder the four corners in the anti-clockwise direction.
    four_corners = np.array([[0, 0],
                             [scope_1st_d - 1, 0],
                             [scope_1st_d - 1, scope_2nd_d - 1],
                             [0, scope_2nd_d - 1]])

    # Find the nearest corner.
    offset_to_corners = four_corners - start_point
    distance_to_corners = np.linalg.norm(np.abs(offset_to_corners), ord=1, axis=1)
    idx = np.argmin(distance_to_corners)
    nearest_corner = four_corners[idx]

    # Move to the nearest corner.
    offset = offset_to_corners[idx]
    offset_1st_d_list = np.arange(1, abs(offset[0]) + 1) * np.sign(offset[0])
    offset_2nd_d_list = np.arange(1, abs(offset[1]) + 1) * np.sign(offset[1])

    if len(offset_1st_d_list) == 0 and len(offset_2nd_d_list) == 0:
        # This copy() function is especially important!!!
        return start_point.copy(), []

    # First traverse the first dimension; then the second dimension.
    traverse_1st_d = start_point + \
        np.column_stack((offset_1st_d_list, np.zeros(len(offset_1st_d_list), dtype=int)))
    path_to_corner = traverse_1st_d.tolist()

    if len(path_to_corner) == 0:
        traverse_2nd_d = start_point + \
                         np.column_stack((np.zeros(len(offset_2nd_d_list), dtype=int), offset_2nd_d_list))
    else:
        traverse_2nd_d = path_to_corner[-1] + \
                         np.column_stack((np.zeros(len(offset_2nd_d_list), dtype=int), offset_2nd_d_list))
    path_to_corner += traverse_2nd_d.tolist()

    return nearest_corner, path_to_corner


def plan_zigzag_path(start_point, scope_1st_d, scope_2nd_d):
    start_corner, path_to_corner = plan_path_to_corner(start_point, scope_1st_d, scope_2nd_d)
    path = path_to_corner

    # First traverse the first dimension; then the second dimension.
    offset_sign_1st_d = 1 if start_corner[0] == 0 else -1
    offset_sign_2nd_d = 1 if start_corner[1] == 0 else -1
    next_position = start_corner

    if start_corner.tolist() == [scope_1st_d - 1, 0] or start_corner.tolist() == [0, scope_2nd_d - 1]:
        for count_2nd_d in range(0, scope_2nd_d):
            for count_1st_d in range(1, scope_1st_d):
                next_position += [offset_sign_1st_d, 0]
                path.append(next_position.tolist())

            offset_sign_1st_d *= -1
            next_position += [0, offset_sign_2nd_d]
            path.append(next_position.tolist())
    else:
        for count_1st_d in range(0, scope_1st_d):
            for count_2nd_d in range(1, scope_2nd_d):
                next_position += [0, offset_sign_2nd_d]
                path.append(next_position.tolist())

            offset_sign_2nd_d *= -1
            next_position += [offset_sign_1st_d, 0]
            path.append(next_position.tolist())

    path = path[:-1]
    return path


def run_policy(env, arg_list, n_pursuers, max_ep_len=None, num_episodes=100, use_seed=True,
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

    offset_action_encoding = {(-1, 0): 0,
                              (1, 0): 1,
                              (0, 1): 2,
                              (0, -1): 3,
                              (0, 0): 4}

    while i_episode < num_episodes:
        if render:
            if not save_frames:
                env.render()
                time.sleep(1e-3)
            else:
                frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))

        # Plan once every episode.
        if ep_len == 0:
            paths = [[] for _ in range(n_pursuers)]
            for agent_id in range(n_pursuers):
                base_env = env.env.env.env
                current_position = (base_env.pursuer_layer.get_position(agent_id)).copy()
                paths[agent_id] = plan_zigzag_path(current_position,
                                                   arg_list.x_size, arg_list.y_size)

        actions = []
        dones = []
        for agent_id in range(n_pursuers):
            o, r, d, _ = env.last_vsos(agent_id)

            # Systematic strategy.
            agents_ep_ret[agent_id] += r

            base_env = env.env.env.env
            current_position = base_env.pursuers[agent_id].current_pos

            if ep_len > len(paths[agent_id]):
                next_position = current_position
            else:
                next_position = paths[agent_id][ep_len]

            offset = next_position - current_position
            a = offset_action_encoding[tuple(offset)]

            actions.append(a)
            dones.append(d)

        for agent_id in range(n_pursuers):
            d, a = dones[agent_id], actions[agent_id]
            if d:
                env.step(None)
            else:
                env.step(a)

            o_new, _, _, _ = env.last_vsos(agent_id)

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
                print('Episode %3d \t EpRet %.3f \t EpLen %d \t CaptureRate %.3f \t Collisions %d \t CollideObstacles %d' %
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


if __name__ == '__main__':
    arg_list = parse_args()

    # Instantiate environment.
    env = pursuit.env(max_cycles=arg_list.episode_len,
                      x_size=arg_list.x_size, y_size=arg_list.y_size,
                      n_evaders=arg_list.n_targets, n_pursuers=arg_list.n_pursuers,
                      obs_range=11,
                      surround=False, n_catch=1,
                      freeze_evaders=True)

    run_policy(env,
               arg_list,
               n_pursuers=arg_list.n_pursuers,
               max_ep_len=arg_list.episode_len,
               num_episodes=arg_list.n_episodes,
               use_seed=arg_list.use_seed,
               render=arg_list.render,
               save_frames=arg_list.save_frames)

    print("SUCCESS!")
