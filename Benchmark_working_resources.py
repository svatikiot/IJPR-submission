import math
import numpy as np
import pandas as pd
from datetime import datetime
import Benchmark_df_creation
import Benchmark_workers_availability


def working_resources(setup_times, loom_sequence, loom_load, workers_list, process_times, number_of_workers):
    # print(f"{datetime.now()}  - Allocating Resources.")
    order_to_add = []
    sorted_looms = []
    # loom_last = {}
    earliest_worker_start = np.zeros((number_of_workers, 1))
    # loom_df.sort_values(['currentLoad'], ascending=[True])
    loom_load = sorted(loom_load.items(), key=lambda x: x[1], reverse=True)
    for i in loom_load:
        sorted_looms.append(i[0])
    for m in sorted_looms:
        loom_load[m] = 0
        if len(loom_sequence[m]) > 0:
            # print(f"{datetime.now()}  - Allocating Resources for jobs on loomID: {m}")
            for i in range(1, len(loom_sequence[m])):
                new_order = loom_sequence[m][i]
                old_order = loom_sequence[m][i - 1]
                setup_time = setup_times[m].loc[old_order, new_order]
                process_time = process_times.loc[new_order, m]
                for worker_resource in range(number_of_workers):
                    earliest_worker_start[worker_resource] = (
                        Benchmark_workers_availability.workers_availability(workers_list[worker_resource][:],
                                                                  loom_load[m],
                                                                  setup_time))
                chosen_worker = np.argmin(earliest_worker_start)
                workers_list[int(chosen_worker)].extend(
                    [int(earliest_worker_start[chosen_worker]),
                     int(earliest_worker_start[chosen_worker] + setup_time)])
                workers_list[int(chosen_worker)].sort()
                # daily_setup_times[math.floor((earliest_worker_start[chosen_worker]) / 1440)] += setup_time
                # else:
                #    daily_setup_times[math.floor(earliest_start / 1440)] +=
                # print(daily_setup_time)
                loom_load[m] = float((earliest_worker_start[chosen_worker]) + setup_time + process_times.loc[new_order, m])
                # loom_last[m] = new_order
                setup_start = (earliest_worker_start[chosen_worker])
                order_to_add.append(
                    Benchmark_df_creation.final_df_creation(new_order, m, setup_time, setup_start, process_time))
                # print(f"{datetime.now()}  - Part of chainID: {new_order} on loomID: {m} done.")
    print (f"Total solution is {max(loom_load)}")
    return order_to_add, max(loom_load)
