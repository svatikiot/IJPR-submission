import pandas as pd
import math
import Benchmark_assignment
import Benchmark_sequence
import Benchmark_working_resources
import time
import json

pd.options.mode.chained_assignment = None  # default='warn'


def heuristic_1():
    final_values = {}
    f = open('N_10_20_30_40_50.json')
    data = json.load(f)

    # loop over all instances in json
    for instance_number in data.keys():
        # reading from instance json
        lower_bound = data[instance_number]['LB']
        number_of_orders = data[instance_number]['Number_of_jobs']
        number_of_machines = data[instance_number]['Number_of_machines']
        if number_of_machines <= 20 and number_of_orders <= 1000:
            alpha = data[instance_number]['Alpha']
            process = pd.DataFrame.from_dict(data[instance_number]['ProcessTimes'])
            process.columns = process.columns.astype(int)
            setups = {}
            for m in data[instance_number]['SetupTimes'].keys():
                setups[int(m)] = pd.DataFrame.from_dict(data[instance_number]['SetupTimes'][m])

            # loop for each worker
            for number_of_workers in [1, 3, 5]:
                instance = str(instance_number) + "-Work" + str(number_of_workers)
                results = [instance_number, instance, number_of_machines, number_of_orders, number_of_workers, alpha,
                           lower_bound]
                final_values[instance] = {}
                final_values[instance]['Number_of_jobs'] = number_of_orders
                final_values[instance]['Number_of_machines'] = number_of_machines
                final_values[instance]['Number_of_workers'] = number_of_workers
                final_values[instance]['Alpha'] = alpha
                final_values[instance]['LB'] = lower_bound
                final_values[instance]['Loom_Sequence'] = {}
                final_values[instance]['Assignments'] = {}
                final_values[instance]['Master_Upper_Bound'] = math.inf
                final_values[instance]['Total_Upper_Bound'] = math.inf
                final_values[instance]['First_Step_Solution'] = math.inf
                for form in ["form2"]:
                    print(
                        f"Instance {instance_number} for Jobs: {number_of_orders}, Machines: {number_of_machines}, Workers: {number_of_workers}, alpha {alpha}, Way:{form}")
                    # setups, process = data_creation.data_creation(number_of_orders, number_of_machines, alpha)
                    master_sol = math.inf
                    total_sol = math.inf
                    assignment_sol = math.inf
                    start = time.time()
                    iterations = 0
                    max_assignments = number_of_orders * number_of_machines

                    while max_assignments >= number_of_orders:
                        workers_list = [[math.inf] for i in range(number_of_workers)]
                        if form == "form1":
                            process_time_assgn, order_dict, max_assignments, first_sol = Benchmark_assignment.gh1_assignment_form1(
                                process, setups, max_assignments)
                        elif form == "form2":
                            process_time_assgn, order_dict, max_assignments, first_sol = Benchmark_assignment.gh1_assignment_form2(
                                process, setups, max_assignments)
                        else:
                            process_time_assgn, order_dict, max_assignments, first_sol = Benchmark_assignment.gh1_assignment_form3(
                                process, setups, max_assignments)

                        if first_sol < final_values[instance]['First_Step_Solution']:
                            final_values[instance]['First_Step_Solution'] = first_sol
                            assignment_sol = first_sol
                        elif first_sol < assignment_sol:
                            assignment_sol = first_sol

                        # print(process_time_assgn, order_dict)

                        # sequence for large aTSP
                        loom_sequence = {}
                        loom_load = {}
                        loom_load_max = 0
                        for m in range(number_of_machines):
                            loom_sequence[m] = []
                            loom_load[m] = 0
                            if len(order_dict[m]) > 0:
                                loom_sequence[m], loom_load[m] = Benchmark_sequence.sequence_dependent_tsp(m, process_time_assgn,
                                                                                                 setups[m])
                                if loom_load[m] > loom_load_max:
                                    loom_load_max = loom_load[m]
                        # print(loom_sequence, loom_load)

                        final_orders, solution = Benchmark_working_resources.working_resources(setups, loom_sequence,
                                                                                            loom_load, workers_list,
                                                                                            process_time_assgn,
                                                                                            number_of_workers)

                        if loom_load_max < final_values[instance]['Master_Upper_Bound']:
                            final_values[instance]['Master_Upper_Bound'] = loom_load_max
                            master_sol = loom_load_max
                            for m in range(number_of_machines):
                                final_values[instance]['Loom_Sequence'][m] = loom_sequence[m]
                                final_values[instance]['Assignments'][m] = process_time_assgn.loc[:, m].values.tolist()
                        elif loom_load_max < master_sol:
                            master_sol = loom_load_max

                        if solution < final_values[instance]['Total_Upper_Bound']:
                            final_values[instance]['Total_Upper_Bound'] = solution
                            total_sol = solution
                            sol_assignments = max_assignments
                            final_values[instance]['Gap'] = (final_values[instance][
                                                                 'Total_Upper_Bound'] - lower_bound) / lower_bound * 100

                        elif solution < total_sol:
                            total_sol = solution
                            sol_assignments = max_assignments

                        # print(final_orders)
                        # print(pd.DataFrame.from_dict(final_orders))

                        max_assignments -= 1
                        iterations += 1
                    end = time.time()
                    gap = ((total_sol - lower_bound) / lower_bound) * 100
                    print(f"Gap is {gap}")

                    results.extend(
                        [assignment_sol, iterations, master_sol, total_sol, gap, end - start])
                results = pd.DataFrame([results])
                with open("Experiments_IJPR_Benchmark.csv", 'a', newline='') as f:
                    results.to_csv(f, header=False, index=False)

    return final_values


def main_benchmark():
    final_values = heuristic_1()

    with open("Final_solutions_1000.json", "w") as outfile:
        json.dump(final_values, outfile)


main_benchmark()
