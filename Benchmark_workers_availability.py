import math
import numpy as np

def workers_availability(work, machines_time, setup_time):
    machines_time += 0.01
    work.append(machines_time)
    # print ("Machines time",machines_time)
    work.sort()
    # print(work)
    # print ("Workers list", workers_list)
    position_of_time = work.index(machines_time)
    # print(machines_time, setup_time)
    # print(work[position_of_time + 1])
    if position_of_time % 2 == 0:
        if machines_time + setup_time < work[position_of_time + 1]:
            machines_best = machines_time
        else:
            found = False
            while position_of_time + 3 < len(work) and not found:
                if work[position_of_time + 2] + setup_time < work[position_of_time + 3]:
                    machines_best = work[position_of_time + 2] + 0.01
                    found = True
                else:
                    position_of_time += 2
    else:
        found = False
        while position_of_time + 2 < len(work) and not found:
            if work[position_of_time + 1] + setup_time < work[position_of_time + 2]:
                machines_best = work[position_of_time + 1] + 0.01
                found = True
            else:
                position_of_time += 2
    # print ("Machines best:",machines_best)
    # print ("Work",work)
    # print ("Machines best", machines_best - 0.01)
    return machines_best - 0.01
