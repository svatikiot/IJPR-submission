import math
import numpy as np
from pyomo.environ import ConcreteModel, ConstraintList, Objective, Var, RangeSet, Binary, NonNegativeReals
from pyomo.opt import SolverFactory
# from pyomo.repn.plugins.ampl.ampl_ import Binary, NonNegativeReals
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import *
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import gurobipy as gp

pd.set_option("display.max_rows", None, "display.max_columns", None)


def gh1_assignment_form1(process_time, setup_time, max_assignments):
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(process_time.columns)
    # print(machine_ids)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(process_time.index)[1:]
    # print(job_ids)
    # quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids,
                                      index=list(process_time.index))  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.Percent = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):

        sum_of_obj_rule = model.C_max

        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    # constraint for maximum number of splits
    constraint_sum = 0
    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Y[i, m] >= model.Percent[i, m])
            constraint_sum += model.Y[i, m]
            #model.constraints.add(
            #    model.Percent[i, m] * process_time.iloc[i, m] >= model.Y[i, m] * max(0, setup_time[int(m)].loc[
            #        setup_time[int(m)][i] > 0, i].min()))
    model.constraints.add(constraint_sum <= max_assignments)
    # print(constraint_sum <= max_assignments)

    # constraints for quantity
    for i in job_ids:
        model.constraints.add((sum(model.Percent[i, m] for m in machine_ids) == 1))
        # print(sum(model.Percent[i, m] for m in machine_ids) == 1)

    # makespan constraint

    for m in machine_ids:
        constraint_sum = 0
        for i in job_ids:
            constraint_sum += model.Percent[i, m] * process_time.iloc[i, m]
        model.constraints.add(constraint_sum <= model.C_max)
        # print(constraint_sum <= model.C_max)
        # print(max(setup_time[int(m)].max()))
    opt = SolverFactory('gurobi')
    opt.options["MIPGap"] = 0.01
    opt.options['timelimit'] = 60
    # model.pprint()
    results = opt.solve(model, tee=False)
    log_infeasible_constraints(model)
    solution = value(model.obj)
    max_assignments = 0
    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:
            if varobject[index].value != 0:
                # print("   ", index, varobject[index].value)
                if str(v) == "Percent" and varobject[index].value > 0.001:
                    # print("   ", index, varobject[index].value)
                    # quantities_assgn.loc[index[0], index[1]] = round(varobject[index].value, 2)
                    process_time_assgn.loc[index[0], index[1]] = round(
                        varobject[index].value * process_time.loc[index[0], index[1]], 2)
                    order_dict[index[1]].append(index[0])

                if str(v) == "Y" and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    max_assignments += 1
    # print(max_assignments, len(process_time.index))

    # print(process_time_assgn, order_dict, max_assignments)

    # calculate load on each machine and find max of setups for LB

    print(f"Assignment Solution is {solution}")

    return process_time_assgn, order_dict, max_assignments, solution


def gh1_assignment_form2(process_time, setup_time, max_assignments):
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(process_time.columns)
    # print(machine_ids)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(process_time.index)[1:]
    # print(job_ids)
    # quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids,
                                      index=list(process_time.index))  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.Percent = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):

        sum_of_obj_rule = model.C_max

        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    # constraint for maximum number of splits
    constraint_sum = 0
    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Y[i, m] >= model.Percent[i, m])
            constraint_sum += model.Y[i, m]
            model.constraints.add(
                model.Percent[i, m] * process_time.iloc[i, m] >= model.Y[i, m] * max(0, setup_time[int(m)].loc[
                    setup_time[int(m)][i] > 0, i].min()))
    model.constraints.add(constraint_sum <= max_assignments)
    # print(constraint_sum <= max_assignments)

    # constraints for quantity
    for i in job_ids:
        model.constraints.add((sum(model.Percent[i, m] for m in machine_ids) == 1))
        # print(sum(model.Percent[i, m] for m in machine_ids) == 1)

    # makespan constraint

    for m in machine_ids:
        constraint_sum = 0
        for i in job_ids:
            constraint_sum += model.Percent[i, m] * process_time.iloc[i, m] + max(0, setup_time[int(m)].loc[
                setup_time[int(m)][i] > 0, i].min()) * model.Y[i, m]
        model.constraints.add(constraint_sum <= model.C_max)
        # print(constraint_sum <= model.C_max)
        # print(max(setup_time[int(m)].max()))
    opt = SolverFactory('gurobi_direct')
    opt.options["MIPGap"] = 0.05
    opt.options['timelimit'] = 60
    # model.pprint()
    results = opt.solve(model, tee=False)
    log_infeasible_constraints(model)
    solution = value(model.obj)
    max_assignments = 0
    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:
            if varobject[index].value != 0:
                # print("   ", index, varobject[index].value)
                if str(v) == "Percent" and varobject[index].value > 0.001:
                    # print("   ", index, varobject[index].value)
                    # quantities_assgn.loc[index[0], index[1]] = round(varobject[index].value, 2)
                    process_time_assgn.loc[index[0], index[1]] = round(
                        varobject[index].value * process_time.loc[index[0], index[1]], 2)
                    order_dict[index[1]].append(index[0])

                if str(v) == "Y" and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    max_assignments += 1
    # print(max_assignments, len(process_time.index))

    # print(process_time_assgn, order_dict, max_assignments)

    # calculate load on each machine and find max of setups for LB

    print(f"Assignment Solution is {solution}")

    return process_time_assgn, order_dict, max_assignments, solution


def gh1_assignment_form3(process_time, setup_time, max_assignments):
    delta = 0.001
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(process_time.columns)
    # print(machine_ids)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(process_time.index)[1:]
    # print(job_ids)
    # quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids,
                                      index=list(process_time.index))  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.Percent = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.X = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.First = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):

        sum_of_obj_rule = model.C_max

        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    constraint_sum = 0
    for i in job_ids:
        for m in machine_ids:
            constraint_sum += model.Y[i, m] + model.X[i, m]
            # model.constraints.add(model.Percent[i,m] >= 0.1)
            # model.constraints.add(model.Y[i, m] <= model.First[i, m])
            model.constraints.add(model.X[i, m] + model.Y[i, m] <= 1)
            model.constraints.add(model.Y[i, m] >= model.Percent[i, m])
            model.constraints.add(model.X[i, m] >= model.First[i, m])
            model.constraints.add(
                model.Percent[i, m] * process_time.iloc[i, m] >= model.Y[i, m] * max(0, setup_time[int(m)].loc[
                    setup_time[int(m)][i] > 0, i].min()))
            # model.constraints.add(model.Percent[i, m] + model.First[i, m] <= 1)
            model.constraints.add(model.X[i, m] - model.First[i, m] <= 1 - delta)
    model.constraints.add(constraint_sum <= max_assignments)

    # constraints for quantity
    for i in job_ids:
        model.constraints.add((sum(model.Percent[i, m] + model.First[i, m] for m in machine_ids) == 1))

    for m in machine_ids:
        model.constraints.add(sum(model.X[i, m] for i in job_ids) == 1)

    # makespan constraint

    for m in machine_ids:
        constraint_sum = 0
        constraint_sum2 = 0
        for i in job_ids:
            constraint_sum2 += model.First[i, m] * process_time.iloc[i, m] - model.X[i, m] * delta
            constraint_sum += model.Percent[i, m] * process_time.iloc[i, m] + max(0, setup_time[int(m)].loc[
                setup_time[int(m)][i] > 0, i].min()) * model.Y[i, m]
        # model.constraints.add(constraint_sum2 <= 1)
        model.constraints.add(constraint_sum + constraint_sum2 <= model.C_max)
        # print(constraint_sum <= model.C_max)
        # print(max(setup_time[int(m)].max()))

    opt = SolverFactory('gurobi')
    opt.options["MIPGap"] = 0.01
    opt.options['timelimit'] = 60
    # model.pprint()
    results = opt.solve(model, tee=False)
    log_infeasible_constraints(model)
    solution = value(model.obj)
    max_assignments = 0
    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:
            if varobject[index].value != 0:
                # print("   ", index, varobject[index].value)
                if (str(v) == "Percent" or str(v) == "First") and varobject[index].value > 0.001:
                    # print("   ", index, varobject[index].value)
                    # quantities_assgn.loc[index[0], index[1]] = round(varobject[index].value, 2)
                    process_time_assgn.loc[index[0], index[1]] = round(
                        varobject[index].value * process_time.loc[index[0], index[1]], 2)
                    order_dict[index[1]].append(index[0])

                if (str(v) == "Y" or str(v) == "X") and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    max_assignments += 1
    # print(max_assignments, len(process_time.index))

    # print(process_time_assgn, order_dict, max_assignments)

    # calculate load on each machine and find max of setups for LB

    print(f"Assignment Solution is {solution}")

    return process_time_assgn, order_dict, max_assignments, solution


'''
def gh2_splitting(orders_new_df, lower_bound):
    orders = orders_new_df.reset_index(drop=False, inplace=False)
    orders_new = []
    for i in range(len(orders)):

        if orders.loc[i, "length_mt"] >= 2 * lower_bound:
            times = math.floor(orders.loc[i, "length_mt"] / lower_bound)
            orders.loc[i, "length_mt"] = orders.loc[i, "length_mt"] / times
            orders_new.extend([orders.loc[i, :].values.tolist()] * times)
        else:
            orders_new.append(orders.loc[i, :].values.tolist())

        # orders_new.append(orders.loc[i, :].values.tolist())

    orders_new_df = pd.DataFrame(orders_new,
                                 columns=['chainType', 'type', "start_date", 'length_mt', 'ca', 'cc', 'chain_id',
                                          'yarns', 'incom', 'comb', 'comb_height', 'strokes_per_meter', 'ybf',
                                          'deliveryDate', 'loomEfficiency'])
    count_parts = {}
    for i in orders_new_df.index:
        # print(order)
        if orders_new_df.loc[i, 'chainType'] in count_parts.keys():
            count_parts[orders_new_df.loc[i, 'chainType']] += 1
        else:
            count_parts[orders_new_df.loc[i, 'chainType']] = 1
    for i in orders_new_df.index:
        if count_parts[orders_new_df.loc[i, 'chainType']] < 10:
            orders_new_df.loc[i, 'chainType_part'] = str(orders_new_df.loc[i, 'chainType']) + '0' + str(
                count_parts[orders_new_df.loc[i, 'chainType']])
        else:
            orders_new_df.loc[i, 'chainType_part'] = str(orders_new_df.loc[i, 'chainType']) + str(
                count_parts[orders_new_df.loc[i, 'chainType']])
        count_parts[orders_new_df.loc[i, 'chainType']] -= 1
    orders_new_df = orders_new_df.set_index('chainType_part')
    return orders_new_df


def gh1_small_dedicated_assignment(machines, jobs, setup_time):
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(machines.index)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(jobs.index)
    quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):

        sum_of_obj_rule = model.C_max

        print(sum_of_obj_rule)
        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    for i in job_ids:
        model.constraints.add((sum(model.Y[i, m] for m in machine_ids) == 1))

    for m in machine_ids:
        constraint_sum = 0
        for i in job_ids:
            # (process time + min_setup) * Y_i
            constraint_sum += \
                (round(jobs.loc[i, 'length_mt'] *
                       jobs.loc[i, 'strokes_per_mt'] * jobs.loc[i, 'ybf'] / (
                               machines.loc[m, 'loomSpeed'] * jobs.loc[i, 'loomEfficiency']), 2) +
                 max(0, setup_time[int(m)].loc[setup_time[int(m)][i] > 0, i].min())) * model.Y[i, m]
        model.constraints.add(constraint_sum + machines.loc[m, "currentLoad"] <= model.C_max)

    opt = SolverFactory('gurobi')
    opt.options["MIPGap"] = 0.05
    opt.options['timelimit'] = 120

    results = opt.solve(model, tee=True)
    # results = opt.solve(model)
    log_infeasible_constraints(model)
    # print(results)
    max_assignments = 0
    print(value(model.obj))
    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:

            if varobject[index].value != 0:
                # print("   ", index, varobject[index].value)
                if str(v) == "Y" and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    quantities_assgn.loc[index[0], index[1]] = jobs.loc[index[0], 'length_mt']
                    process_time_assgn.loc[index[0], index[1]] = round(
                        jobs.loc[index[0], 'length_mt'] * jobs.loc[index[0], 'strokes_per_mt'] * jobs.loc[
                            index[0], 'ybf'] / (
                                machines.loc[
                                    index[1], 'loomSpeed'] * jobs.loc[index[0], 'loomEfficiency']), 2)
                    order_dict[index[1]].append(index[0])
    # print("PROCESS TIMES", process_time_assgn)
    # print(machines.currentLoad)
    return quantities_assgn, process_time_assgn, order_dict


def gh1_large_dedicated_assignment(machines, jobs, setup_time, max_assignments, lower_bound):
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(machines.index)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(jobs.index)
    quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.Q = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):

        sum_of_obj_rule = model.C_max

        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    # constraint for maximum number of splits
    constraint_sum = 0
    for i in job_ids:
        for m in machine_ids:
            constraint_sum += model.Y[i, m]
    model.constraints.add(
        (constraint_sum) <= max_assignments)
    # constraints for quantity

    for i in job_ids:
        model.constraints.add(
            (sum(model.Q[i, m] for m in machine_ids) == jobs.loc[i, "length_mt"]))
        # print(sum(model.Q[i, m] for m in machine_ids) == jobs.loc[i, "targetMeters"])

    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Q[i, m] >= model.Y[i, m] * min(jobs.loc[i, "length_mt"], lower_bound))
            model.constraints.add(model.Q[i, m] <= model.Y[i, m] * jobs.loc[i, "length_mt"])
            # print(model.Q[i, m] <= model.Y[i, m] * jobs.loc[i, 'targetMeters'])
            # print(model.Q[i, m] >= model.Y[i, m] * min(jobs.loc[i, 'targetMeters'], 50))

    # makespan constraint

    for m in machine_ids:
        constraint_sum = 0
        for i in job_ids:
            # (process time + min_setup) * Y_i
            constraint_sum += model.Q[i, m] * round(
                jobs.loc[i, 'strokes_per_mt'] * jobs.loc[i, 'ybf'] / (
                        machines.loc[m, 'loomSpeed'] * jobs.loc[i, 'loomEfficiency']), 2) + \
                              max(0, setup_time[int(m)].loc[setup_time[int(m)][i] > 0, i].min()) * model.Y[i, m]
        model.constraints.add(constraint_sum + machines.loc[m, "currentLoad"] <= model.C_max)

    opt = SolverFactory('gurobi')
    opt.options["MIPGap"] = 0.05
    opt.options['timelimit'] = 120


    results = opt.solve(model, tee=True)
    # results = opt.solve(model)
    log_infeasible_constraints(model)
    # print(results)
    max_assignments = 0
    print(value(model.obj))
    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:

            if varobject[index].value != 0:
                # print("   ", index, varobject[index].value)
                if str(v) == "Q" and varobject[index].value > 0.1:
                    # print("   ", index, varobject[index].value)
                    quantities_assgn.loc[index[0], index[1]] = round(varobject[index].value, 2)
                    process_time_assgn.loc[index[0], index[1]] = round(
                        varobject[index].value * jobs.loc[index[0], 'strokes_per_mt'] * jobs.loc[index[0], 'ybf'] / (
                                machines.loc[
                                    index[1], 'loomSpeed'] * jobs.loc[index[0], 'loomEfficiency']), 2)
                    order_dict[index[1]].append(index[0])

                if str(v) == "Y" and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    max_assignments += 1
    # print("PROCESS TIMES", process_time_assgn)
    # print(machines.currentLoad)
    return quantities_assgn, process_time_assgn, order_dict, max_assignments

'''


def gh1_assignment_completion_times_obj(machines, jobs, setup_time, max_assignments, lower_bound):
    V = math.inf
    order_dict = {}  # dictionary that will contain all orders with keys machine ids
    machine_ids = list(machines.index)
    for m in machine_ids:
        order_dict[m] = []  # initialize empty lists in dict
    job_ids = list(jobs.index)
    quantities_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # quantity of each order on each loom
    process_time_assgn = pd.DataFrame(0, columns=machine_ids, index=job_ids)  # process time of each order on each loom

    # Initialize model

    model = ConcreteModel()
    model.jobs = job_ids
    model.machines = machine_ids
    model.Y = Var(model.jobs, model.machines, within=Binary, initialize=0.0)
    model.Q = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.X = Var(model.jobs, model.jobs, model.machines, within=Binary, initialize=0.0)
    model.C_m = Var(model.jobs, model.machines, within=NonNegativeReals, initialize=0.0)
    model.C = Var(model.jobs, within=NonNegativeReals, initialize=0.0)

    # model.C_max = Var(within=NonNegativeReals, initialize=0.0)

    def obj_rule(model):
        sum_of_obj_rule = 0
        for f in job_ids:
            sum_of_obj_rule += model.C[f]
        # print(sum_of_obj_rule)
        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    # constraint for maximum number of splits
    constraint_sum = 0
    for i in job_ids:
        for m in machine_ids:
            constraint_sum += model.Y[i, m]
    model.constraints.add(
        (constraint_sum) <= max_assignments)
    # constraints for quantity

    for i in job_ids:
        model.constraints.add(
            (sum(model.Q[i, m] for m in machine_ids) == jobs.loc[i, "length_mt"]))
        # print(sum(model.Q[i, m] for m in machine_ids) == jobs.loc[i, "targetMeters"])

    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Q[i, m] >= model.Y[i, m] * min(jobs.loc[i, "length_mt"], lower_bound))
            model.constraints.add(model.Q[i, m] <= model.Y[i, m] * jobs.loc[i, "length_mt"])
            # print(model.Q[i, m] <= model.Y[i, m] * jobs.loc[i, 'targetMeters'])
            # print(model.Q[i, m] >= model.Y[i, m] * min(jobs.loc[i, 'targetMeters'], 50))
    '''
    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Y[i, m] >= model.Q[i, m] / jobs.loc[i, "length_mt"])
    '''

    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Y[i, m] == sum(model.X[i, j, m]
                                                       for j in job_ids if i != j))

    for j in job_ids:
        for m in machine_ids:
            model.constraints.add(model.Y[j, m] == sum(
                model.X[i, j, m] for i in job_ids if i != j))

    for m in machine_ids:
        for i in job_ids:
            model.constraints.add(
                model.C_m[i, m] >= model.Y[i, m] * (setup_time[int(m)].loc[setup_time[int(m)][i] > 0, i].min() +
                                                    machines.loc[m, 'currentLoad']) + model.Q[i, m] * round(
                    jobs.loc[i, 'strokes_per_mt'] * jobs.loc[i, 'ybf'] / (
                            machines.loc[m, 'loomSpeed'] * jobs.loc[i, 'loomEfficiency']), 2))

    for i in job_ids:
        for j in job_ids:
            if i != j:
                for m in machine_ids:
                    model.constraints.add(model.C_m[j, m] - model.C_m[i, m] + V * (
                            1 - model.X[i, j, m]) >= model.Y[j, m] * (setup_time[int(m)].loc[i, j] +
                                                                      machines.loc[
                                                                          m, 'currentLoad']) +
                                          model.Q[j, m] * round(
                        jobs.loc[j, 'strokes_per_mt'] * jobs.loc[j, 'ybf'] / (
                                machines.loc[m, 'loomSpeed'] * jobs.loc[j, 'loomEfficiency']), 2))
    for i in job_ids:
        for m in machine_ids:
            model.constraints.add(model.C[i] >= model.C_m[i, m])
    '''
    for m in machine_ids:
        constraint_sum = 0
        for i in job_ids:
            # (process time + min_setup) * Y_i
            constraint_sum += model.Q[i, m] * round(
                jobs.loc[i, 'strokes_per_mt'] * jobs.loc[i, 'ybf'] / (
                        machines.loc[m, 'loomSpeed'] * jobs.loc[i, 'loomEfficiency']), 2) + \
                              setup_time[int(m)].loc[setup_time[int(m)][i] > 0, i].min() * model.Y[i, m]
        model.constraints.add(constraint_sum + machines.loc[m, "currentLoad"] <= model.C_max)
    '''
    opt = SolverFactory('gurobi')
    opt.options["MIPGap"] = 0.05
    # opt.options['timelimit'] = 120

    ''' model.pprint()
    '''
    results = opt.solve(model, tee=True)
    # results = opt.solve(model)
    log_infeasible_constraints(model)
    # print(results)
    max_assignments = 0
    print(value(model.obj))
    for v in model.component_objects(Var, active=True):
        print("Variable component object", v)
        print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:

            if varobject[index].value != 0:
                print("   ", index, varobject[index].value)
                if str(v) == "Q" and varobject[index].value > 0.1:
                    # print("   ", index, varobject[index].value)
                    quantities_assgn.loc[index[0], index[1]] = round(varobject[index].value, 2)
                    process_time_assgn.loc[index[0], index[1]] = round(
                        varobject[index].value * jobs.loc[index[0], 'strokes_per_mt'] * jobs.loc[index[0], 'ybf'] / (
                                machines.loc[
                                    index[1], 'loomSpeed'] * jobs.loc[index[0], 'loomEfficiency']), 2)
                    order_dict[index[1]].append(index[0])

                if str(v) == "Y" and varobject[index].value > 0.9:
                    # print("   ", index, varobject[index].value)
                    max_assignments += 1
    print(max_assignments)
    print("PROCESS TIMES", process_time_assgn)
    print(machines.currentLoad)
    return quantities_assgn, process_time_assgn, order_dict, max_assignments
