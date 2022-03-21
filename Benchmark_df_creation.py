import math
import numpy as np
import pandas as pd
from datetime import datetime


def final_df_creation(chainID, m, setup_time, setup_start, process_time):
    # add to orders_final_df
    order_to_add = {}
    order_to_add['orderID'] = int(chainID)  # chainID
    order_to_add['loomID'] = int(m)  # loomID
    order_to_add['setupTime'] = float(setup_time)  # setupTime
    order_to_add['setupStartTime'] = float(setup_start)  # setupStartTime
    order_to_add['setupEndTime'] = float(setup_start + setup_time)  # setupEndTime
    order_to_add['processingTime'] = float(process_time)  # processingTime
    order_to_add['processStartTime'] = float(setup_start + setup_time)  # processingStartTime
    order_to_add['processEndTime'] = float(setup_start + setup_time + process_time)  # processingEndTime
    return order_to_add
