"""
Modify / Extend Openai spinningup to multi-agent RL.
"""
import argparse
import numpy as np
import time

import torch

from pursuit_game import pursuit_vstp as pursuit

from actor_critic.trainer.actor_critic_trainer import ActorCriticAgents
from actor_critic.trainer.run_utils import setup_logger_kwargs

from actor_critic.common.utils import count_vars
from actor_critic.common.logx import EpochLogger
from actor_critic.common.mpi_pytorch import setup_pytorch_for_mpi
from actor_critic.common.mpi_tools import mpi_fork, proc_id, num_procs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_ep_len", type=int, default=500)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    # parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--pi_lr", type=float, default=1e-4)
    # parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--vf_lr", type=float, default=1e-4)
    parser.add_argument("--train_v_iters", type=int, default=80)
    parser.add_argument("--save_freq", type=int, default=100)

    parser.add_argument("--n_pursuers", type=int, default=4)
    parser.add_argument("--n_targets", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default='stp_actor_critic_4p_1t_6x6_reward_10_1e_4')

    return parser.parse_args()


def train(arg_list):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    n_processes = num_procs()

    # Set up logger and save configuration.
    logger_kwargs = setup_logger_kwargs(arg_list.exp_name, arg_list.seed)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(arg_list)

    # Instantiate environment.
    env = pursuit.env(max_cycles=500,
                      x_size=6, y_size=6,
                      n_evaders=arg_list.n_targets, n_pursuers=arg_list.n_pursuers,
                      obs_range=11,
                      surround=True, n_catch=4,
                      freeze_evaders=True)

    # Random seed.
    seed = arg_list.seed + 10000 * (proc_id() + 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    observation_space = env.env.env.env.observation_space[0]
    action_space = env.env.env.env.action_space[0]
    obs_dim = np.prod(observation_space.shape)
    act_dim = action_space.n

    agents = ActorCriticAgents(arg_list.n_pursuers, obs_dim, act_dim, arg_list.steps_per_epoch,
                               arg_list.gamma, arg_list.lam, arg_list.pi_lr, arg_list.vf_lr, arg_list.train_v_iters)
    # Sync params across processes
    agents.syn_params()
    # Set up model saving.
    logger.setup_pytorch_saver(agents.model)

    # Count variables.
    var_counts = tuple(count_vars(module) for module in [agents.model.pi, agents.model.v])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Prepare for interaction with environment.
    start_time = time.time()
    env.reset()
    ep_ret, ep_len, ep_collisions, collisions_with_obstacles = 0, 0, 0, 0
    agents_ep_ret = [0 for _ in range(arg_list.n_pursuers)]

    # Main loop: collect experience in env and update/log each epoch.
    for epoch in range(arg_list.epochs):
        for t in range(arg_list.steps_per_epoch):
            actions = []
            dones = []
            for agent_id in range(arg_list.n_pursuers):

                o, r, d, _ = env.last_vsos(agent_id)
                o = np.reshape(o, -1)
                agents_ep_ret[agent_id] += r

                a, v, logp = agents.model.step(torch.as_tensor(o, dtype=torch.float32))

                actions.append(a)
                dones.append(d)

                # Save and log.
                agents.buffer.store(o, a, r, v, logp, agent_id)
                logger.store(VVals=v)

            for agent_id in range(arg_list.n_pursuers):
                d, a = dones[agent_id], actions[agent_id]
                if d:
                    env.step(None)
                else:
                    env.step(a)

            ep_len += 1
            ep_collisions += env.env.env.env.n_collision_events_per_multiagent_step

            timeout = (ep_len == arg_list.max_ep_len)
            terminal = env.env.env.env.is_terminal or timeout
            epoch_ended = (t == (arg_list.steps_per_epoch - 1))

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print("Warning: trajectory cut off by epoch at %d steps." % ep_len, flush=True)

                # If trajectory didn't reach terminal state, bootstrap value target.
                if timeout or epoch_ended:
                    v_list = []
                    for agent_id in range(arg_list.n_pursuers):
                        o, r, d, _ = env.last_vsos(agent_id)
                        o = np.reshape(o, -1)
                        _, v, _ = agents.model.step(torch.as_tensor(o, dtype=torch.float32))
                        v_list.append(v)
                else:
                    v_list = [0 for _ in range(arg_list.n_pursuers)]

                agents.buffer.finish_path(v_list)

                if terminal:
                    # Only save EpRet / EpLen if trajectory finished.
                    ep_ret = np.mean(agents_ep_ret)
                    capture_rate = sum(env.env.env.env.evaders_gone) / len(env.env.env.env.evaders_gone)
                    collisions_with_obstacles = env.env.env.env.n_collision_with_obstacles
                    logger.store(EpRet=ep_ret, EpLen=ep_len, CaptureRate=capture_rate,
                                 Collisions=ep_collisions, CollideObstacles=collisions_with_obstacles)

                env.reset()
                ep_ret, ep_len, ep_collisions, collisions_with_obstacles = 0, 0, 0, 0
                agents_ep_ret = [0 for _ in range(arg_list.n_pursuers)]

        # Save model.
        if (epoch % arg_list.save_freq == 0) or (epoch == arg_list.epochs - 1):
            logger.save_state({"env": env}, itr=epoch)

        # Perform update.
        loss_pi_old, loss_v_old, delta_loss_pi, delta_loss_v, kl, entropy = agents.update()
        logger.store(LossPi=loss_pi_old,
                     LossV=loss_v_old,
                     DeltaLossPi=delta_loss_pi,
                     DeltaLossV=delta_loss_v,
                     KL=kl,
                     Entropy=entropy)

        # Log info about epoch.
        # xaxis.
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("MAEnvIterations", (epoch + 1) * arg_list.steps_per_epoch)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * arg_list.steps_per_epoch * arg_list.n_pursuers)
        # yaxis.
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("CaptureRate", with_min_and_max=True)
        logger.log_tabular("Collisions", with_min_and_max=True)
        logger.log_tabular("CollideObstacles", with_min_and_max=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        # legend.
        logger.log_tabular("Processes", n_processes)
        logger.log_tabular("Agents", arg_list.n_pursuers)
        logger.dump_tabular()


if __name__ == "__main__":
    arg_list = parse_args()
    # Run parallel code with MPI.
    mpi_fork(arg_list.cpu)
    train(arg_list)
    print('DONE!')
