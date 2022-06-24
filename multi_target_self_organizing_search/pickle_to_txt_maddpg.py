import argparse
import pickle
import os.path as osp


def main(sub_folder):
    data_path = "./results/sos_MADDPG/learning_curves_s8/"
    output_file_name = osp.join(data_path, "progress.txt")
    exp_name = "maddpg_sos_experiment"
    # Number of iters per save
    iters_per_step = 1

    csv_rows = [["Episode", "EpLen", "EpRet", "CaptureRate", "Collisions", "CollisionsWithObstacles", "QLoss", "PLoss", "HaveLoss", "Time(s)"]]
    rew_file_name = osp.join(data_path, exp_name + '_rewards.pkl')
    length_file_name = osp.join(data_path, exp_name + '_lengths.pkl')
    capture_file_name = osp.join(data_path, exp_name + '_capture_rates.pkl')
    collision_file_name = osp.join(data_path, exp_name + '_collisions.pkl')
    collisions_with_obstacles_file_name = osp.join(data_path, exp_name + '_collisions_with_obstacles.pkl')
    q_loss_file_name = osp.join(data_path, exp_name + '_q_loss.pkl')
    p_loss_file_name = osp.join(data_path, exp_name + '_p_loss.pkl')
    have_loss_or_not_file_name = osp.join(data_path, exp_name + '_have_loss_or_not.pkl')
    time_file_name = osp.join(data_path, exp_name + '_time.pkl')

    with open(rew_file_name, 'rb') as f_rew, \
         open(length_file_name, 'rb') as f_length,  \
         open(capture_file_name, 'rb') as f_capture, \
         open(collision_file_name, 'rb') as f_collision, \
         open(collisions_with_obstacles_file_name, 'rb') as f_collisions_with_obstacles, \
         open(q_loss_file_name, 'rb') as f_q_loss, \
         open(p_loss_file_name, 'rb') as f_p_loss, \
         open(have_loss_or_not_file_name, 'rb') as f_have_loss_or_not, \
         open(time_file_name, 'rb') as f_time:

        ep_rewards = pickle.load(f_rew)
        episode_lengths = pickle.load(f_length)
        episode_capture_rates = pickle.load(f_capture)
        episode_collisions = pickle.load(f_collision)
        episode_collisions_with_obstacles = pickle.load(f_collisions_with_obstacles)
        episode_q_losses = pickle.load(f_q_loss)
        episode_p_losses = pickle.load(f_p_loss)
        episode_have_loss_or_not = pickle.load(f_have_loss_or_not)
        episode_times = pickle.load(f_time)

        for i, (rew, length, capture, collision, collisions_with_obstacles, q_loss, p_loss, have_loss_or_not, time) in \
                enumerate(zip(ep_rewards,
                              episode_lengths,
                              episode_capture_rates,
                              episode_collisions,
                              episode_collisions_with_obstacles,
                              episode_q_losses,
                              episode_p_losses,
                              episode_have_loss_or_not,
                              episode_times)):

            csv_rows.append([str(i * iters_per_step),
                             str(length), str(rew), str(capture), str(collision), str(collisions_with_obstacles),
                             str(q_loss), str(p_loss), str(have_loss_or_not), str(time)])

    with open(output_file_name, 'w') as fp:
        fp.write("\n".join(["\t".join(row) for row in csv_rows]))

    print('Write to file:', output_file_name)


if __name__ == "__main__":
    sub_folders = []
    for seed in range(1):
        sub_folder = 'learning_curves_s' + str(seed)
        main(sub_folder)
    # main()
    print('SUCCESS!')
