The maddpg codes are from
- https://github.com/parametersharingmadrl/parametersharingmadrl
- https://github.com/openai/maddpg

# of agents: 8
observation shape: 11 * 11 * 3 -> flatten
# of actions: 5

actor/policy function input shape: 11 * 11 * 3 + 1 = 364
critic/Q function input shape: (364 + 5) * 8 = 2952


From train_sos_maddpg.py -> test_sos_maddpg.py:
- num-episodes: 1
- restore: False -> True
- display: False -> True
- save-dir: policy -> test_policy
- plots-dir: learning_curves -> testing_curves.
- save-rate: 1
- comment:
    - collect experience;
    - update trainers;
    - U.save_state(arg_list.save_dir, saver=saver).
- add and modify the following to save render results:
    import PIL
    from datetime import datetime
    parser.add_argument("--save-frames", default=True)

    frame_list = []

    if arg_list.display:
        if arg_list.save_frames:
            frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        else:
            time.sleep(0.1)
            env.env.sisl_env.render()

    if done or terminal:
        if arg_list.save_frames:
            data_path = arg_list.save_dir
            now = datetime.now()
            time_format = "%Y%m%d%H%M%S"
            timestamp = now.strftime(time_format)
            filename = data_path + "/pursuit_" + \
                       str("{:.3f}".format(episode_rewards)) + \
                       str("{:.3f}".format(episode_step)) + \
                       str("{:.3f}".format(capture_rate)) + \
                       str("{:.3f}".format(collisions)) + "_" + \
                       + timestamp + ".gif"
            images_to_gif(frame_list, filename)

        obs_n = env.reset()
        episode_step = 0
        episode_rewards.append(0)
        for a in agent_rewards:
            a.append(0)

    if not arg_list.save_frames:
        env.env.sisl_env.close()  # EnvSOS

    def images_to_gif(image_list, output_name):
        output_dir = osp.join(data_path, 'pngs')
        os.system("mkdir " + output_dir)
        for idx, im in enumerate(image_list):
            im.save(output_dir + "/target" + str(idx) + ".png")

        os.system("/usr/local/bin/ffmpeg -i " + output_dir + "/target%d.png " + output_name)
        os.system("rm -r " + output_dir)
        print("Write to file:", output_name)

- add and modify the following path and seed:

    data_path = "/Users/lijunsun/Workspace/selforganizing_search_pursuit/results/sos_maddpg/policy_s0/"

    def parse_args(seed):
        parser.add_argument("--save-dir", type=str, default=data_path, help="directory in which training state and model should be saved")

    - replace:
        seed = arg_list.seed + 10000
      with
        seed = arg_list.seed

