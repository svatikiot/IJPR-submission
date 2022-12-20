from __future__ import division
import sys
import random
import time
import math
import cplex
from cplex.exceptions import CplexSolverError
from docplex.cp.model import CpoModel
import json
from csv import reader

# Scheduling unrelated machines with job splitting, setup resources and sequence dependency
# Avgerinos I., Mourtos, I., Vatikiotis, S., Zois, G.
# Algorithm 2: A Logic-Based Benders Decomposition Algorithm

print("---------------------------------------------------------------------------------------------------")

file1 = open('dataset.json') # The dataset parameters are imported.
data1 = json.load(file1)
file2 = open('solutions.json') # The solutions of the GHA are imported.
data2 = json.load(file2)
file_name = "Results.txt" # The output file

instances1 = []
for key in data1:
	instances1.append(key)

for instance in instances1:
	with open(file_name, 'a') as output:
		for key2 in data2:
			if instance in key2:
				updated_key = key2
				start_time = time.time()
				for key in data1[instance]:
					if key == "Number_of_jobs":
						jobs = int(data1[instance][key])+1 # The number of jobs J is defined.
					if key == "Number_of_machines":
						machines = data1[instance][key] # The number of machines M is defined.
				processing_times = [[0.0 for m in range(machines)] for i in range(jobs)] # Initialization of the processing times
				setup_times = [[[0.0 for m in range(machines)] for j in range(jobs)] for i in range(jobs)] # Initialization of the setup times
				for key in data1[instance]:
					if key == "ProcessTimes":
						for m in data1[instance][key]:
							for i in range(jobs):
								processing_times[i][int(m)] = round(data1[instance][key][m][i], 2) # The processing times are imported.
					if key == "SetupTimes":
						for m in data1[instance][key]:
							for i in range(jobs):
								for j in range(jobs):
									setup_times[i][j][int(m)] = data1[instance][key][m][i][j] # The setup times are imported.
				for key in data2[updated_key]:
					if key == "Number_of_workers":
						workers = data2[updated_key][key] # The number of workers R is defined
					if key == "LB":
						l_bound = float(data2[updated_key][key]) # The lower bound of GHA is imported.
					if key == "Total_Upper_Bound":
						big_M = math.ceil(data2[updated_key][key])*2 # The big-M value (referred as 'V' in the text) is determined by the upper bound of GHA.
						best_bound = big_M

				#fixed_var are the values that the variables would receive, considering the solution of GHA.
				fixed_x = [[[0.0 for m in range(machines)] for j in range(jobs)] for i in range(jobs)] 
				fixed_y = [[0.0 for m in range(machines)] for i in range(jobs)]
				fixed_W = [[0.0 for m in range(machines)] for i in range(jobs)]
				actual_p = [[0.0 for m in range(machines)] for i in range(jobs)]
				fixed_n = [[0.0 for m in range(machines)] for i in range(jobs)]
				quantity = [100*(i != 0) for i in range(jobs)] # 'quantity[i]' is fixed to 100% for all jobs i\in J* and 0% for job 0
				for key in data2[updated_key]:
					if key == "Loom_Sequence":
						for m in data2[updated_key][key]:
							order = 1
							for i in range(1, len(data2[updated_key][key][m])):
								fixed_x[int(data2[updated_key][key][m][i-1])][int(data2[updated_key][key][m][i])][int(m)] = 1.0
								fixed_n[int(data2[updated_key][key][m][i])][int(m)] = order
								order = order + 1
							fixed_x[int(data2[updated_key][key][m][i])][0][int(m)] = 1.0
					if key == "Assignments":
						for m in data2[updated_key][key]:
							for i in range(jobs):
								actual_p[i][int(m)] = data2[updated_key][key][m][i]
				for i in range(1, jobs):
					for m in range(machines):
						if any(fixed_x[j][i][m] == 1.0 for j in range(jobs)):
							fixed_y[i][m] = 1.0
				for i in range(jobs):
					for m in range(machines):
						if i > 0:
							fixed_W[i][m] = round(actual_p[i][m]/processing_times[i][m], 2)
						else:
							fixed_W[i][m] = 0.0
				for m in range(machines):
					fixed_n[0][m] = jobs-1
				u_bound = 0
				for m in range(machines):
					machine_makespan = 0
					for n in range(1, jobs):
						for i in range(1, jobs):
							if fixed_n[i][m] == n:
								for j in range(jobs):
									if fixed_x[j][i][m] == 1.0:
										machine_makespan = machine_makespan + setup_times[j][i][m] + fixed_W[i][m]*processing_times[i][m]
					if machine_makespan > u_bound:
						u_bound = machine_makespan #'u_bound' is the objective value of the solution of GHA.

				print("Instance "+str(updated_key))
				master = cplex.Cplex()
				master.parameters.mip.tolerances.integrality.set(0.0) #The tolerance of integrality is set to absolute 0
				master.set_results_stream(None)
				master.set_warning_stream(None)
				master.parameters.timelimit.set(600) #The time limit of the master problem is set to 600 seconds.

				master_obj = []
				master_lb = []
				master_ub = []
				master_types = []
				master_names = []

				VARS = []
				VALUES = []

				x = [[["x_"+str(i)+","+str(j)+","+str(m) for m in range(machines)] for j in range(jobs)] for i in range(jobs)]
				for i in range(jobs):
					for j in range(jobs):
						for m in range(machines):
							if i != j:
								master_obj.append(0.0)
								master_lb.append(0.0)
								master_ub.append(1.0)
								master_types.append("B")
								master_names.append(x[i][j][m])
							if i == j:
								master_obj.append(0.0)
								master_lb.append(0.0)
								master_ub.append(0.0)
								master_types.append("B")
								master_names.append(x[i][j][m])
							VARS.append(x[i][j][m])
							VALUES.append(fixed_x[i][j][m])
				y = [["y_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
				for i in range(jobs):
					for m in range(machines):
						master_obj.append(0.0)
						master_lb.append(0.0)
						master_ub.append(1.0)
						master_types.append("B")
						master_names.append(y[i][m])
						VARS.append(y[i][m])
						VALUES.append(fixed_y[i][m])
				W = [["W_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
				for i in range(jobs):
					for m in range(machines):
						master_obj.append(0.0)
						master_lb.append(0.0)
						master_ub.append(quantity[i])
						master_types.append("I")
						master_names.append(W[i][m])
						VARS.append(W[i][m])
						VALUES.append(quantity[i]*fixed_W[i][m])
				n = [["n_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
				for i in range(jobs):
					for m in range(machines):
						master_obj.append(0.0)
						master_lb.append(0.0)
						master_ub.append(jobs-1) #Constraints (21)
						master_types.append("I")
						master_names.append(n[i][m])
						VARS.append(n[i][m])
						VALUES.append(fixed_n[i][m])
				z = ["z_0"]
				master_obj.append(1.0)
				master_lb.append(l_bound)
				master_ub.append(big_M)
				master_types.append("C")
				master_names.append(z[0])
				VARS.append(z[0])
				VALUES.append(u_bound)

				master.variables.add(obj = master_obj,
									 lb = master_lb,
									 ub = master_ub,
									 types = master_types,
									 names = master_names)
				master.MIP_starts.add([VARS, VALUES], master.MIP_starts.effort_level.solve_MIP) # The warm start solution is supplied to the solver.
				master_expressions = []
				master_senses = []
				master_rhs = []

				for i in range(1, jobs):
					#Constraints (15)
					constraint = cplex.SparsePair(ind = [W[i][m] for m in range(machines)],
										  		  val = [1.0 for m in range(machines)])
					master_expressions.append(constraint)
					master_senses.append("E")
					master_rhs.append(quantity[i])
					for m in range(machines):
						#Constraints (16)
						constraint = cplex.SparsePair(ind = [y[i][m]]     + [W[i][m]],
											  		  val = [quantity[i]] + [-1.0])
						master_expressions.append(constraint)
						master_senses.append("G")
						master_rhs.append(0.0)
						#Constraints (17)
						constraint = cplex.SparsePair(ind = [y[i][m]] + [x[i][j][m] for j in range(jobs)],
											 		  val = [1.0]     + [-1.0 for j in range(jobs)])
						master_expressions.append(constraint)
						master_senses.append("E")
						master_rhs.append(0.0)
						#Constraints (18)
						constraint = cplex.SparsePair(ind = [y[i][m]] + [x[j][i][m] for j in range(jobs)],
											 		  val = [1.0]     + [-1.0 for j in range(jobs)])
						master_expressions.append(constraint)
						master_senses.append("E")
						master_rhs.append(0.0)
						for j in range(jobs):
							if i != j:
								#Constraints (20)
								constraint = cplex.SparsePair(ind = [n[i][m]] + [n[j][m]] + [x[i][j][m]],
													  		  val = [1.0]     + [-1.0]    + [jobs])
								master_expressions.append(constraint)
								master_senses.append("L")
								master_rhs.append(jobs-1)
				for m in range(machines):
					#Constraints (19)
					constraint = cplex.SparsePair(ind = [x[0][i][m] for i in range(jobs)],
										  		  val = [1.0 for i in range(jobs)])
					master_expressions.append(constraint)
					master_senses.append("L")
					master_rhs.append(1.0)
					#Constraints (22)
					objective_function = cplex.SparsePair(ind = [z[0]] + [W[i][m] for i in range(1, jobs)]                               + [x[j][i][m] for j in range(jobs) for i in range(1, jobs)],
												  		  val = [1.0]  + [-(processing_times[i][m]/quantity[i]) for i in range(1, jobs)] + [-setup_times[j][i][m] for j in range(jobs) for i in range(1, jobs)])
					master.linear_constraints.add(lin_expr = [objective_function], 
												  senses = ["G"],
												  rhs = [0.0])

				master.linear_constraints.add(lin_expr = master_expressions, 
											  senses = master_senses,
											  rhs = master_rhs)

				master.objective.set_sense(master.objective.sense.minimize)
				master.solve()

				convergence = False #Convergence binary parameter
				iteration = 0 #Iteration k

				print("---------------------------------------------------------------------------------------")
				print("Iteration		Lower Bound		Upper Bound		Time")
				print("---------------------------------------------------------------------------------------")
				while(convergence == False):
					iteration = iteration + 1
					lower_bound = round(master.solution.MIP.get_best_objective(), 2) # The lower bound of the master problem
					used_machines = [] # Set \hat{M} is defined.
					for m in range(machines):
						if any(master.solution.get_values(y[i][m]) > 0.9 for i in range(1, jobs)):
							used_machines.append(m)
					machine_seq = []
					load = []
					for m in range(len(used_machines)):
						machine_seq.append([])
						load.append([])
						start_job = 0
						next_job = jobs
						while(next_job != 0):
							for i in range(jobs):
								if master.solution.get_values(x[start_job][i][used_machines[m]]) > 0.9:
									if i != 0:
										machine_seq[m].append(i)
										load[m].append(master.solution.get_values(W[i][used_machines[m]])/quantity[i])

									next_job = i
									start_job = i
									break

					subproblem = CpoModel()
					setup = {}
					for m in range(len(used_machines)):
						for i in range(len(machine_seq[m])):
							if i == 0:
								start = (0, big_M)
								end = (0, big_M)
								size = int(setup_times[0][machine_seq[m][i]][used_machines[m]])
								setup[(m, i)] = subproblem.interval_var(start, end, size, name = "setup"+str(used_machines[m])+","+str(machine_seq[m][i]))
							else:
								start = (0, big_M)
								end = (0,big_M)
								size = int(setup_times[machine_seq[m][i-1]][machine_seq[m][i]][used_machines[m]])
								setup[(m, i)] = subproblem.interval_var(start, end, size, name = "setup"+str(used_machines[m])+","+str(machine_seq[m][i]))

					makespan = [subproblem.integer_var(0, big_M, name = "makespan")]

					#Constraints (23)
					subproblem.add(sum([subproblem.pulse(setup[(m, i)], 1) for m in range(len(used_machines)) for i in range(len(machine_seq[m]))]) <= workers)
					for m in range(len(used_machines)):
						for i in range(1, len(machine_seq[m])):
							#Constraints (24)
							subproblem.add(subproblem.start_of(setup[(m, i)]) >= subproblem.end_of(setup[(m, i-1)]) + load[m][i-1]*processing_times[machine_seq[m][i-1]][used_machines[m]])
					for m in range(len(used_machines)):
						for i in range(len(machine_seq[m])):
							#Constraints (25)
							subproblem.add(makespan[0] >= subproblem.end_of(setup[(m, i)]) + load[m][i]*processing_times[machine_seq[m][i]][used_machines[m]])

					total_cost = makespan[0]
					subproblem.add(subproblem.minimize(total_cost))

					sol = subproblem.solve(TimeLimit = 60, trace_log = False) #An arbitrary time limit for the subproblem is set; this time limit is never reached.

					if sol: #The solution 'sol' of the subproblem is always feasible.
						zeta = 0
						for m in range(len(used_machines)):
							for i in range(len(machine_seq[m])):
								machine_makespan = round(sol[setup[(m, i)]][1] + load[m][i]*(processing_times[machine_seq[m][i]][used_machines[m]]), 2)
							if machine_makespan > zeta:
								zeta = machine_makespan
						upper_bound = round(zeta, 2) #The upper bound of iteration is defined.
						if best_bound > upper_bound:
							best_bound = upper_bound

					end_time = time.time()
					if (upper_bound - lower_bound)/lower_bound < 0.01: #Maximum gap E is set to 1% (0.01)
						convergence = True #If E is reached, Convergence is set to True
						print(str(iteration)+"				"+str(lower_bound)+"			"+str(upper_bound)+"			"+str(round(end_time - start_time, 2)))
						break
					else:
						print(str(iteration)+"				"+str(lower_bound)+"			"+str(upper_bound)+"			"+str(round(end_time - start_time, 2)))
						assignments = [] #Set A is defined.
						for m in range(len(used_machines)):
							for i in range(len(machine_seq[m])):
								if i == 0:
									assignments.append([0, machine_seq[m][i], used_machines[m]])
								elif i == len(machine_seq[m]) - 1:
									assignments.append([machine_seq[m][i], 0, used_machines[m]])
								else:
									assignments.append([machine_seq[m][i-1], machine_seq[m][i], used_machines[m]])
						total_assignments = len(assignments)

						pairs = [] #Set P is defined
						splits = []
						for i in range(1, jobs):
							for m in range(machines):
								if master.solution.get_values(W[i][m]) > 0.9:
									pairs.append([i, m])
									splits.append(master.solution.get_values(W[i][m]))

						cut_obj = []
						cut_lb = []
						cut_ub = []
						cut_types = []
						cut_names = []
							
						rho = []
						for i in range(len(pairs)):
							rho.append("rho_"+str(i)+","+str(iteration))
							cut_obj.append(0.0)
							cut_lb.append(0.0)
							cut_ub.append(1.0)
							cut_types.append("B")
							cut_names.append(rho[i])
						greater = [] #'greater' stands for variables \lamda^{>=}_{p}
						for i in range(len(pairs)):
							greater.append(">_"+str(i)+","+str(iteration))
							cut_obj.append(0.0)
							cut_lb.append(0.0)
							cut_ub.append(1.0)
							cut_types.append("B")
							cut_names.append(greater[i])
						less = [] #'less' stands for variables \lamda^{<=}_{p}
						for i in range(len(pairs)):
							less.append("<_"+str(i)+","+str(iteration))
							cut_obj.append(0.0)
							cut_lb.append(0.0)
							cut_ub.append(1.0)
							cut_types.append("B")
							cut_names.append(less[i])

						master.variables.add(obj = cut_obj,
											 lb = cut_lb,
											 ub = cut_ub,
											 types = cut_types,
											 names = cut_names)

						cut_expressions = []
						cut_senses = []
						cut_rhs = []

						for i in range(len(pairs)):
							#Constraints (30)
							cut = cplex.SparsePair(ind = [W[pairs[i][0]][pairs[i][1]]] + [greater[i]],
												   val = [1.0]                         + [-quantity[pairs[i][0]]]) #'quantity' is an adequately big number 'V'
							cut_expressions.append(cut)
							cut_senses.append("L")
							cut_rhs.append(splits[i] - 0.01) #0.01 is the value of small number 'v'
							#Constraints (29)
							cut = cplex.SparsePair(ind = [W[pairs[i][0]][pairs[i][1]]] + [less[i]],
												   val = [-1.0]                        + [-quantity[pairs[i][0]]])
							cut_expressions.append(cut)
							cut_senses.append("L")
							cut_rhs.append(-splits[i] - 0.01)
							#Constraints (31)
							cut = cplex.SparsePair(ind = [rho[i]] + [greater[i]] + [less[i]],
												   val = [1.0]    + [-1.0]       + [-1.0])
							cut_expressions.append(cut)
							cut_senses.append("G")
							cut_rhs.append(-1.0)
						#Constraints (32)
						cut = cplex.SparsePair(ind = [z[0]] + [x[assignments[a][0]][assignments[a][1]][assignments[a][2]] for a in range(len(assignments))] + [rho[i] for i in range(len(pairs))],
										       val = [1.0]  + [-zeta for a in range(len(assignments))]                                                      + [-zeta for i in range(len(pairs))])
						cut_expressions.append(cut)
						cut_senses.append("G")
						cut_rhs.append(zeta - (total_assignments + len(pairs))*zeta) #'total_assignments' stands for |A|, 'len(pairs' stands for |P|

						master.linear_constraints.add(lin_expr = cut_expressions, 
													  senses = cut_senses,
													  rhs = cut_rhs)

						master.objective.set_sense(master.objective.sense.minimize)
						master.solve()

						if iteration == 20: #'K' limit of iterations is set to 20.
							convergence = True
					print("---------------------------------------------------------------------------------------")
					output.write(str(updated_key)+";"+str(lower_bound)+";"+str(best_bound)+";"+str(round(end_time - start_time, 2))+";"+str(iteration)+"\n") #The results are exported to the output file.
					output.close()

