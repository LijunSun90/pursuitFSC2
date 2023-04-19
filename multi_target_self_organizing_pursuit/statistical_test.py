import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_log_folder", type=str, default="data")

    return parser.parse_args()


def statistical_test():

    all_args = parse_args()

    # Get data.

    # clustering_no_memory
    # hard_clustering
    # random_clustering
    # fish_schooling_search
    # sop_40_40_8_2_
    # sop_40_40_16_4_v1
    # sop_40_40_16_4_
    # sop_40_40_32_8_
    # sop_40_40_64_16_
    # sop_40_40_128_32_
    # sop_40_40_256_64_
    # sop_40_40_512_128_
    # sop_40_40_1024_256_

    data_file_1 = os.path.join(all_args.data_log_folder, "sop_40_40_16_4_.txt")
    # data_file_2 = os.path.join(all_args.data_log_folder,
    #                            "ablation_study",
    #                            "hard_clustering",
    #                            "sop_40_40_1024_256_.txt")
    # data_file_2 = os.path.join(all_args.data_log_folder,
    #                            "log_mappo",
    #                            "sop_mappo",
    #                            "sop_40_40_16_4_Epoch2000x14.txt")
    data_file_2 = os.path.join(all_args.data_log_folder,
                               "log_mappo",
                               "sop_ippo",
                               "sop_40_40_16_4_.txt")
    # data_file_2 = os.path.join(all_args.data_log_folder,
    #                            "log_actor_critic",
    #                            "sop_actor_critic",
    #                            "sop_40_40_16_4_Epoch2000x26.txt")

    print("data_file_1:", data_file_1)
    print("data_file_2:", data_file_2)

    data_1 = pd.read_table(data_file_1)
    data_2 = pd.read_table(data_file_2)

    # EpLen	CaptureRate	Collisions

    # 1.

    metric_1 = data_1['EpLen'][:-1]
    metric_2 = data_2['EpLen'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('EpLen:', result_pvalue)

    # 2

    metric_1 = data_1['CaptureRate'][:-1]
    metric_2 = data_2['CaptureRate'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('CaptureRate:', result_pvalue)

    # 3

    metric_1 = data_1['Collisions'][:-1]
    metric_2 = data_2['Collisions'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('Collisions:', result_pvalue)

    pass


if __name__ == "__main__":
    statistical_test()
    print("COMPLETE!")

