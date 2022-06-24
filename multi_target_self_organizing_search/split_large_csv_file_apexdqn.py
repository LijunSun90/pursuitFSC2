import pandas as pd
import os
import os.path as osp
import time


def main():
    # data_path = "./results/sos_Apex-DQN/APEX_pursuit_a26cf_00000_0_2022-05-03_22-43-29/"
    # data_path = "./results/stp_ray_a2c/A2C/A2C_pursuit_87738_00000_0_2022-05-24_15-01-11/"
    data_path = "./results/"
    for folder, sub_folders, files in os.walk(data_path):
        if 'progress.csv' in files:
            process_data_file(folder)


def process_data_file(data_path):
    file_name = osp.join(data_path, "progress.csv")
    output_file_name = osp.join(data_path, "progress.txt")

    columns = ["episodes_total", "training_iteration",
               "episode_reward_mean", "episode_len_mean", "time_total_s",
               "custom_metrics/capture_rate_mean", "custom_metrics/collisions_mean",
               "custom_metrics/collisions_with_obstacles_mean"]

    dataset = []

    # chunksize parameter specifies the number of rows per chunk.
    # (The last chunk may contain fewer than chunksize rows.)
    # 4: Time 94.095 s = 1.568 min.
    # 3: Time 73.191 s = 1.220 min.
    # 2: Time 75.404 s = 1.257 min
    chunksize = 10 ** 3
    with pd.read_csv(file_name, chunksize=chunksize) as reader:
        for chunk in reader:
            chunk_data = chunk[columns]
            # process(chunk)
            dataset.append(chunk_data)

    dataset = pd.concat(dataset)
    print(dataset)
    dataset.to_csv(output_file_name, sep='\t')
    print("Write to file:", output_file_name)


if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print("Time {:.3f} s = {:.3f} min.".format(duration, duration / 60))
    print("SUCCESS!")
