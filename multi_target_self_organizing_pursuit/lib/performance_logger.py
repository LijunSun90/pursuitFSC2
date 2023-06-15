"""
Author: Lijun Sun.
Date: Tue 21 Feb 2023.
"""
import os
import time
import numpy as np


class PerformanceLogger:

    def __init__(self, log_folder, log_file=None):

        self.log_folder = log_folder

        if log_file is None:
            self.log_file = os.path.join(self.log_folder, "experiment" + time.strftime("%Y%m%d%H%M%S") + ".txt")
        else:
            self.log_file = log_file

        self.episode_performance = dict()

        self.epoch_performance = dict()

        self.epoch_counter = 0

        self.log_row_head = ["epoch"]

        pass

    def create_log_file(self):

        os.makedirs(self.log_folder, exist_ok=True)

        with open(self.log_file, 'a') as output_file:

            output_file.write("\t".join(map(str, self.log_row_head)) + "\n")
            output_file.flush()

        print("=> Create:", self.log_file)

    def reset_episode_performance(self):

        for key in self.episode_performance.keys():

            self.episode_performance[key] = 0

    def reset_epoch_performance(self):

        for key in self.epoch_performance.keys():

            self.epoch_performance[key] = []

    def update_episode_performance(self, key, value):

        if key in self.episode_performance.keys():

            self.episode_performance[key] += value

        else:

            self.episode_performance[key] = value

            self.epoch_performance[key] = []

            self.log_row_head.append(key)

    def end_episode_performance(self):

        for key in self.episode_performance.keys():

            self.epoch_performance[key].append(self.episode_performance[key])

        # Update.

        self.reset_episode_performance()

    def update_epoch_performance(self, key, value):

        if key in self.epoch_performance.keys():

            self.epoch_performance[key].append(value)

        else:

            self.epoch_performance[key] = [value]

            self.log_row_head.append(key)

    def log_dump_epoch_performance(self, epoch_counter=None, is_print=True):

        if epoch_counter is not None:
            self.epoch_counter = epoch_counter

        if not os.path.exists(self.log_file):

            self.create_log_file()

        # Calculate.

        row = [self.epoch_counter]

        for key in self.epoch_performance.keys():

            row.append(np.mean(self.epoch_performance[key]))

        # Write.

        with open(self.log_file, 'a') as output_file:

            output_file.write("\t".join(map(str, row)) + "\n")
            output_file.flush()

        # Print.

        if is_print:

            self.print_epoch_performance()

        # Update.

        if epoch_counter is None:
            self.epoch_counter += 1

        self.reset_epoch_performance()

        pass

    def print_epoch_performance(self):

        print("epoch: {:4f}".format(self.epoch_counter), end=", ")

        for key, value in self.epoch_performance.items():

            print(key + ": {:4f}".format(np.mean(value)), end=", ")

        print()


def example_usage():

    logger = PerformanceLogger("data")

    for idx_epoch in range(3):

        start_epoch_time = time.time()

        for idx_episode in range(5):

            for idx_timestep in range(10):

                logger.update_episode_performance(key="metric", value=1.0)

            if idx_episode < 4:

                logger.update_episode_performance("episode_counter", idx_episode)
                logger.end_episode_performance()

            else:

                logger.reset_episode_performance()

        logger.update_epoch_performance(key="epoch_time_s", value=time.time() - start_epoch_time)

        logger.log_dump_epoch_performance()

    pass


if __name__ == "__main__":

    example_usage()

    print('COMPLETE!')

