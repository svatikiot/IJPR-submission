import math
import numpy as np
from pyomo.environ import ConcreteModel, ConstraintList, Objective, Var, RangeSet, Binary, NonNegativeReals
from pyomo.opt import SolverFactory
# from pyomo.repn.plugins.ampl.ampl_ import Binary, NonNegativeReals
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import value
import pandas as pd
from datetime import datetime
import itertools


def sequence_dependent_tsp(loom, process_time_on_loom, setup_times):
    machine_final_assignment = []
    orderlist = []
    # print(process_time_on_loom.index)
    for i in process_time_on_loom.index:
        # print(i, loom, process_time_on_loom.loc[i, loom])
        if process_time_on_loom.loc[i, loom] > 0:
            orderlist.append(i)
        # print(orderlist)
    orderlist.insert(0, 0)  # list of orders with last order on loom
    orderlist2 = orderlist[1:]  # only new orders
    # print(orderlist)
    # print(orderlist2)
    # Initialize model
    model = ConcreteModel()
    model.N = RangeSet(len(orderlist))
    model.X = Var(orderlist, orderlist2, within=Binary, initialize=0.0)
    model.U = Var(orderlist)
    '''
    upper_bound = 0
    lower_bound = 0
    for i in orderlist[1:]:
        upper_bound += process_time_on_loom.loc[i, loom] + setup_times.loc[i, i - 1]
        lower_bound += process_time_on_loom.loc[i, loom]
    '''
    def obj_rule(model):
        sum_of_obj_rule = 0

        for i in orderlist:
            for j in orderlist2:
                if i != j and j != orderlist[0]:
                    sum_of_obj_rule += (model.X[i, j]) * (process_time_on_loom.loc[j, loom] + setup_times.loc[i, j])

        # print(sum_of_obj_rule)
        return sum_of_obj_rule

    model.obj = Objective(rule=obj_rule)
    model.constraints = ConstraintList()

    for i in orderlist[:len(orderlist) - 1]:
        model.constraints.add(sum(model.X[i, j] for j in orderlist2 if i != j) == 1)

    for i in orderlist2:
        model.constraints.add(sum(model.X[j, i] for j in orderlist if i != j) == 1)

    for i in orderlist:
        for j in orderlist:
            if i != j and j != orderlist[0]:
                model.constraints.add(
                    model.U[i] - model.U[j] + model.X[i, j] * len(orderlist) <= len(orderlist) - 1)
                # print((model.U[i] - model.U[j] + model.X[i, j] * len(orderlist) <= len(orderlist) - 1))
    '''
    objective_fun = 0
    for i in orderlist:
        for j in orderlist2:
            if i != j and j != orderlist[0]:
                objective_fun += (model.X[i, j]) * (process_time_on_loom.loc[j, loom] + setup_times.loc[i, j])
    model.constraints.add(objective_fun <= upper_bound)
    model.constraints.add(objective_fun >= first_sol)
    '''
    opt = SolverFactory('gurobi_direct')
    opt.options['timelimit'] = 60
    # opt.setParam('MIPGap', 0.9)
    opt.options["MIPGap"] = 0.005
    # opt.options["MIPGap"] = 0.05
    # opt.setParam('MIPGap', 0.9)
    # results = opt.solve(model, load_solutions=False, tee=True)
    results = opt.solve(model, tee=False)
    # print(results)

    log_infeasible_constraints(model)
    # print("VALUE IS", value(model.obj))
    # print("----------------------------------------------------")
    opt_makespan = value(model.obj)

    for v in model.component_objects(Var, active=True):
        # print("Variable component object", v)
        # print("Type of component object: ", str(type(v))[1:-1])  # Stripping <> for nbconvert
        varobject = getattr(model, str(v))
        # print("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
        for index in varobject:
            if varobject[index].value > 0.7:
                if str(v) == "X":
                    # print("   ", index, varobject[index].value)
                    machine_final_assignment.append(index[0])
                    machine_final_assignment.append(index[1])

    # print(machine_final_assignment)

    # print(machine_final_assignment)

    kappa = 1
    while kappa < len(machine_final_assignment):
        zeta = kappa + 1
        found = False
        # print(zeta, kappa, machine_final_assignment)
        while zeta < len(machine_final_assignment) and not found:
            if machine_final_assignment[kappa] == machine_final_assignment[zeta]:
                machine_final_assignment.insert(kappa + 1, machine_final_assignment[zeta + 1])
                del [machine_final_assignment[zeta + 2]]
                del [machine_final_assignment[zeta + 1]]
                found = True
            zeta += 1
        kappa += 1
    '''
    print(machine_final_assignment)
    for item in machine_final_assignment:
        if machine_final_assignment.count(item) == 1 and item != 0:
            ind = machine_final_assignment.index(item)
            fixed_machine_final_assignment = [machine_final_assignment[ind - 1], item]
            k = machine_final_assignment[ind - 1]
            del [machine_final_assignment[ind - 1]]
            machine_final_assignment.remove(item)
    print(machine_final_assignment)
    print(fixed_machine_final_assignment)
    # print(k)
    while k != 0:
        zeta = 1
        found = False
        while not found:
            print (zeta)
            if machine_final_assignment.count(machine_final_assignment[zeta]) == 1:
                k = machine_final_assignment[zeta - 1]
                del [machine_final_assignment[zeta]]
                del [machine_final_assignment[zeta - 1]]
                fixed_machine_final_assignment.insert(0, k)
                found = True
            else:
                zeta += 1
            # print(k)
    # print(fixed_machine_final_assignment)

    # change in machine order !Attention
    # print(machine_final_assignment)
    # print(jobs)
    # print(setup_times)
    # change order after tsp without increasing makespan
    
    
    
    
    
    delivery_date_list = jobs.loc[:, "deliveryDate"].sort_values()
    for i in jobs.index:
        if i not in machine_final_assignment:
            delivery_date_list = delivery_date_list.drop([i])

    tardiness_sorted = (list(delivery_date_list.index))

    tardiness_sorted.insert(0, orderlist[0])
    print(machine_final_assignment)
    new_makespan = 0
    for i in range(1, len(tardiness_sorted)):
        print(setup_times)
        print(setup_times.loc[
                  tardiness_sorted[i - 1], tardiness_sorted[i]], process_time_on_loom.loc[tardiness_sorted[i], loom])
        new_makespan += setup_times.loc[
                  tardiness_sorted[i - 1], tardiness_sorted[i]] + process_time_on_loom.loc[tardiness_sorted[i], loom]
        #print(delivery_date_list)
    print (new_makespan)
    print (opt_makespan)
    if new_makespan == opt_makespan:
        machine_final_assignment = tardiness_sorted.copy()
    #print (machine_final_assignment)
    best_delivery_date_factor = 0
    for combo in itertools.permutations(tardiness_sorted, len(tardiness_sorted)):
        combo = list(combo)
        new_makespan = 0
        combo.insert(0, orderlist[0])
        # print(combo)
        for i in range(1, len(combo)):
            new_makespan += setup_times.loc[
                                combo[i - 1], combo[i]] + process_time_on_loom.loc[
                                combo[i], loom]
        # print (combo, new_makespan)
        if new_makespan <= opt_makespan:
            # calculate delivery date factor
            delivery_date_factor = 0
            weight = 1
            for k in combo[1:]:
                delivery_date_factor += jobs.loc[k, "deliveryDate"] * weight
                weight += 1
            # print(delivery_date_factor, combo, new_makespan)
            if delivery_date_factor > best_delivery_date_factor:
                best_delivery_date_factor = delivery_date_factor
                machine_final_assignment = combo.copy()
    #print(machine_final_assignment)
    '''
    # print(f"for Machine: {loom}, sequence value is {opt_makespan} and assignment is {machine_final_assignment}")
    return machine_final_assignment, opt_makespan
