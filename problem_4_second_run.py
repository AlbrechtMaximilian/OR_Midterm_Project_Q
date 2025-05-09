import os, time, math
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def load_instance_data(file_path):
    xls = pd.ExcelFile(file_path)
    df_demand = pd.read_excel(xls, "Demand")
    df_intransit = pd.read_excel(xls, "In-transit")
    df_shipcost = pd.read_excel(xls, "Shipping cost")
    df_invcost = pd.read_excel(xls, "Inventory cost")
    df_params = pd.read_excel(xls, "Parameters")

    N, T, J = df_demand.shape[0], df_demand.shape[1] - 2, 3
    D, I, I_0, V = np.zeros((N, T)), np.zeros((N, T)), np.zeros(N), np.zeros(N)
    C = {"H": np.zeros(N), "P": np.zeros(N), "V": np.zeros((N, J)), "F": np.zeros(J), "C": 0}
    V_C, T_lead = 30, np.zeros(J)

    for i in range(N):
        I_0[i] = df_demand.iloc[i, 1]
        for t in range(T):
            D[i, t] = df_demand.iloc[i, t + 2]
            I[i, t] = df_intransit.iloc[i, t + 1]
        V[i] = df_shipcost.iloc[i, 4]
        C["V"][i, 0] = df_shipcost.iloc[i, 1]
        C["V"][i, 1] = df_shipcost.iloc[i, 2]
        C["V"][i, 2] = 0
        C["P"][i] = df_invcost.iloc[i, 2]
        C["H"][i] = df_invcost.iloc[i, 3]

    for _, row in df_params.iterrows():
        param = row["Parameter"]
        if param == "ContainerCost":
            C["C"] = row["Value"]
        elif param == "ContainerVolume":
            V_C = row["Value"]
        elif "FixedCost_Method" in param:
            C["F"][int(param[-1]) - 1] = row["Value"]
        elif "LeadTime_Method" in param:
            T_lead[int(param[-1]) - 1] = int(row["Value"])

    return N, T, J, D, I, I_0, V, V_C, T_lead, C

def solve_ip(N, T, J, D, I, I_0, V, V_C, T_lead, C):
    t_ip_start = time.time()
    # Provided parameters (already read from the Excel file)
    # N: number of products, T: number of time periods, J: number of shipping methods
    # D: demand, I: in-transit inventory, C: cost parameters, V: volume, T_lead: lead times, V_C: container volume

    # Create the Gurobi model
    model = gp.Model("InventoryManagement")

    # Set error parameter
    model.setParam('MIPGap', 0.0)

    # no prints
    model.setParam('OutputFlag', 0)

    # Define sets
    S_I = range(N)  # Products i in {0,  ..., N-1}
    S_T = range(T)  # Time periods t in {0, ..., T-1}
    S_J = range(J)  # Shipping methods j in {0, ..., J-1}

    # Variables
    x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")  # Order quantity x_ijt
    v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")  # Ending inventory v_it
    y = model.addVars(S_J, S_T, vtype=GRB.BINARY, name="y")  # Binary for shipping method y_jt
    z = model.addVars(S_T, vtype=GRB.INTEGER, name="z")  # Number of containers z_t

    # Objective function (1)
    # Holding cost + (Purchasing cost + Variable shipping cost + Fixed shipping cost) + Container cost
    holding_cost = gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T)
    purchasing_and_shipping_cost = gp.quicksum(
        (C["P"][i] + C["V"][i, j]) * x[i, j, t]
        for i in S_I for j in S_J for t in S_T
    ) + gp.quicksum(C["F"][j] * y[j, t] for t in S_T for j in S_J)
    container_cost = gp.quicksum(C["C"] * z[t] for t in S_T)

    model.setObjective(holding_cost + purchasing_and_shipping_cost + container_cost, GRB.MINIMIZE)

    # Constraints
    # Inventory balance (2)
    #J_in_inventory = np.array([1, 2, 3, 3, 3, 3])
    #J_in_inventory = np.array([1, 2, 3, 3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3,3,3])
    J_in_inventory = np.array([min(t + 1, 3) for t in range(T)])

    for i in S_I:
        for t in S_T:
            # Compute the in-transit quantity arriving at time t
            in_inventory = 0
            for j in range(J_in_inventory[t]):
                in_inventory += x[i, j, t - T_lead[j] + 1]
            # Add the constraint for inventory balance
            if t == 0:
                model.addConstr(v[i, t] == in_inventory + I_0[i] + I[i, t] - D[i, t], name=f"InvBalance_{i}_{t}")
            else:
                model.addConstr(v[i, t] == v[i, t - 1] + in_inventory + I[i, t] - D[i, t], name=f"InvBalance_{i}_{t}")
                model.addConstr(v[i, t - 1] >= D[i, t], name=f"Demand_{i}_{t}")

    # Relate order quantity and shipping method (4)
    M = sum(sum(D[i, t] for t in S_T) for i in S_I)  # Large number M as per problem statement
    for j in S_J:
        for t in S_T:
            model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t], name=f"ShippingMethod_{j}_{t}")

    # Container constraint (5)
    for t in S_T:
        model.addConstr(
            gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t],
            name=f"Container_{t}"
        )

    # Non-negativity and binary constraints (6)
    for i in S_I:
        for j in S_J:
            for t in S_T:
                model.addConstr(x[i, j, t] >= 0, name=f"NonNeg_x_{i}_{j}_{t}")
    for i in S_I:
        for t in S_T:
            model.addConstr(v[i, t] >= 0, name=f"NonNeg_v_{i}_{t}")
    for j in S_J:
        for t in S_T:
            model.addConstr(y[j, t] >= 0, name=f"Binary_y_{j}_{t}")  # Already binary due to vtype
    for t in S_T:
        model.addConstr(z[t] >= 0, name=f"NonNeg_z_{t}")

    # Optimize the model
    model.optimize()

    # Print the solution
    """if model.status == GRB.OPTIMAL:
        print("\nOptimal objective value:", model.objVal)
        print("\nOrder quantities (x_ijt):")

        for t in S_T:
            for i in S_I:
                for j in S_J:
                    if x[i, j, t].x > 0:
                        print(f"x[{i + 1},{j + 1},{t + 1}] = {x[i, j, t].x}")  # +1 to make the index consistent
        print("\nEnding inventory (v_it):")
        for t in S_T:
            for i in S_I:
                if v[i, t].x > 0:
                    print(f"v[{i + 1},{t + 1}] = {v[i, t].x}")
        print("\nShipping method usage (y_jt):")
        for t in S_T:
            for j in S_J:
                if y[j, t].x > 0:
                    print(f"y[{j + 1},{t + 1}] = {y[j, t].x}")
        print("\nNumber of containers (z_t):")
        for t in S_T:
            if z[t].x > 0:
                print(f"z[{t + 1}] = {z[t].x}")
    else:
        print("No optimal solution found.")"""
    t_ip = time.time() - t_ip_start
    return model.objVal, t_ip

def solve_lp_relaxation(N, T, J, D, I, I_0, V, V_C, T_lead, C):
    t_lr_start = time.time()
    model = gp.Model("InventoryManagement_LP")

    # Relaxed precision
    model.setParam('MIPGap', 0.0)

    # no prints
    model.setParam('OutputFlag', 0)

    S_I = range(N)
    S_T = range(T)
    S_J = range(J)

    # Variables
    x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")   # Order quantity
    v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")        # Inventory
    y = model.addVars(S_J, S_T, vtype=GRB.CONTINUOUS, name="y")        # Relaxed binary
    z = model.addVars(S_T, vtype=GRB.CONTINUOUS, name="z")             # Relaxed integer

    # Objective
    holding_cost = gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T)
    purchasing_and_shipping_cost = gp.quicksum(
        (C["P"][i] + C["V"][i, j]) * x[i, j, t]
        for i in S_I for j in S_J for t in S_T
    ) + gp.quicksum(C["F"][j] * y[j, t] for j in S_J for t in S_T)
    container_cost = gp.quicksum(C["C"] * z[t] for t in S_T)

    model.setObjective(holding_cost + purchasing_and_shipping_cost + container_cost, GRB.MINIMIZE)

    # Constraints
    for i in S_I:
        for t in S_T:
            in_inventory = 0
            for j in S_J:
                if t - T_lead[j] + 1 >= 0:
                    in_inventory += x[i, j, t - T_lead[j] + 1]
            if t == 0:
                model.addConstr(v[i, t] == I_0[i] + I[i, t] + in_inventory - D[i, t])
            else:
                model.addConstr(v[i, t] == v[i, t - 1] + I[i, t] + in_inventory - D[i, t])
                model.addConstr(v[i, t - 1] >= D[i, t])

    # Link shipping method usage to order quantities
    M = sum(D[i, t] for i in S_I for t in S_T)
    for j in S_J:
        for t in S_T:
            model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t])

    # Container usage constraint
    for t in S_T:
        model.addConstr(
            gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t]
        )

    # Non-negativity
    for i in S_I:
        for j in S_J:
            for t in S_T:
                model.addConstr(x[i, j, t] >= 0)
    for i in S_I:
        for t in S_T:
            model.addConstr(v[i, t] >= 0)
    for j in S_J:
        for t in S_T:
            model.addConstr(y[j, t] >= 0)
    for t in S_T:
        model.addConstr(z[t] >= 0)

    model.optimize()

    """if model.status == GRB.OPTIMAL:
        print("\nOptimal (LP) objective value:", model.objVal)
    else:
        print("No optimal LP solution found.")"""
    t_lr = time.time()-t_lr_start
    return model.ObjVal, t_lr


def solve_lp_round_fix(N, T, J, D, I, I_0, V, V_C, T_lead, C):
    t_heur_start = time.time()
    # === STEP 1: LP relaxation ===
    model_lp = gp.Model("LP_Relaxation")
    model_lp.setParam('OutputFlag', 0)

    S_I, S_T, S_J = range(N), range(T), range(J)

    x = model_lp.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")
    v = model_lp.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")
    y = model_lp.addVars(S_J, S_T, vtype=GRB.CONTINUOUS, name="y")
    z = model_lp.addVars(S_T, vtype=GRB.CONTINUOUS, name="z")

    # Objective
    model_lp.setObjective(
        gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T) +
        gp.quicksum((C["P"][i] + C["V"][i, j]) * x[i, j, t] for i in S_I for j in S_J for t in S_T) +
        gp.quicksum(C["F"][j] * y[j, t] for j in S_J for t in S_T) +
        gp.quicksum(C["C"] * z[t] for t in S_T),
        GRB.MINIMIZE
    )

    M = sum(D[i, t] for i in S_I for t in S_T)
    for i in S_I:
        for t in S_T:
            in_inventory = gp.quicksum(
                x[i, j, t - T_lead[j] + 1]
                for j in S_J if t - T_lead[j] + 1 >= 0
            )
            if t == 0:
                model_lp.addConstr(v[i, t] == I_0[i] + I[i, t] + in_inventory - D[i, t])
            else:
                model_lp.addConstr(v[i, t] == v[i, t - 1] + I[i, t] + in_inventory - D[i, t])
                model_lp.addConstr(v[i, t - 1] >= D[i, t])

    for j in S_J:
        for t in S_T:
            model_lp.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t])
    for t in S_T:
        model_lp.addConstr(gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t])

    model_lp.optimize()

    # === STEP 2: Fix rounded y and z ===
    fixed_yz = {
        (j, t): 1 if y[j, t].X > 0 else 0
        for j in S_J for t in S_T
    }
    fixed_z = {
        t: int(round(z[t].X))
        for t in S_T
    }

    # === STEP 3: Second LP with fixed y/z ===
    model_fix = gp.Model("Fix_YZ")
    model_fix.setParam('OutputFlag', 0)

    x2 = model_fix.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")
    v2 = model_fix.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")

    model_fix.setObjective(
        gp.quicksum(C["H"][i] * v2[i, t] for i in S_I for t in S_T) +
        gp.quicksum((C["P"][i] + C["V"][i, j]) * x2[i, j, t] for i in S_I for j in S_J for t in S_T) +
        gp.quicksum(C["F"][j] * fixed_yz[j, t] for j in S_J for t in S_T) +
        gp.quicksum(C["C"] * fixed_z[t] for t in S_T),
        GRB.MINIMIZE
    )

    for i in S_I:
        for t in S_T:
            in_inventory = gp.quicksum(
                x2[i, j, t - T_lead[j] + 1]
                for j in S_J if t - T_lead[j] + 1 >= 0
            )
            if t == 0:
                model_fix.addConstr(v2[i, t] == I_0[i] + I[i, t] + in_inventory - D[i, t])
            else:
                model_fix.addConstr(v2[i, t] == v2[i, t - 1] + I[i, t] + in_inventory - D[i, t])
                model_fix.addConstr(v2[i, t - 1] >= D[i, t])

    for j in S_J:
        for t in S_T:
            model_fix.addConstr(gp.quicksum(x2[i, j, t] for i in S_I) <= M * fixed_yz[j, t])
    for t in S_T:
        model_fix.addConstr(gp.quicksum(V[i] * x2[i, 2, t] for i in S_I) <= V_C * fixed_z[t])

    model_fix.optimize()

    t_heur = time.time() - t_heur_start
    return model_fix.ObjVal, t_heur

def solve_naive(N, T, J, D, I, I_0, V, V_C, T_lead, C):
    import time
    t0 = time.time()

    total_cost = 0.0
    inventory = I_0.copy()

    for t in range(T):
        shipping_cost = 0.0
        purchase_cost = 0.0
        ordered_something = False

        for i in range(N):
            available = inventory[i] + I[i, t]
            demand = D[i, t]

            # Naive: no storage – order just enough for this period
            order = max(0, demand - available)
            used_inventory = min(available, demand)
            leftover = available - used_inventory

            if order > 0:
                ordered_something = True
                shipping_cost += C["V"][i, 0] * order
                purchase_cost += C["P"][i] * order

            # No inventory carried over to next period
            inventory[i] = 0

        if ordered_something:
            shipping_cost += C["F"][0]

        total_cost += shipping_cost + purchase_cost

    t1 = time.time()
    return total_cost, t1 - t0


def run_experiments(folder_path):
    from time import time
    results = []

    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".xlsx") or not file.startswith("scenario_"):
            continue

        file_path = os.path.join(folder_path, file)
        scenario_id = int(file.split("_")[1])
        instance_id = int(file.split("_")[3].split(".")[0])

        # Load instance data
        N, T, J, D, I, I_0, V, V_C, T_lead, C = load_instance_data(file_path)

        # Solve using IP
        obj_ip, t_ip = solve_ip(N, T, J, D, I, I_0, V, V_C, T_lead, C)

        # Solve using LP Relaxation
        obj_lr, t_lr = solve_lp_relaxation(N, T, J, D, I, I_0, V, V_C, T_lead, C)

        # Solve using LP Round-and-Fix Heuristic
        obj_heur, t_heur = solve_lp_round_fix(N, T, J, D, I, I_0, V, V_C, T_lead, C)

        # Solve using Naive Heuristic (currently mocked)
        obj_naive, t_naive = solve_naive(N, T, J, D, I, I_0, V, V_C, T_lead, C)

        # Calculate gaps
        gap_lr = (obj_lr - obj_ip) / obj_ip
        gap_heur = (obj_heur - obj_ip) / obj_ip
        gap_naive = (obj_naive - obj_ip) / obj_ip

        # Store result
        results.append({
            "ScenarioID": scenario_id,
            "InstanceID": instance_id,
            "IP_ObjVal": obj_ip,
            "LR_ObjVal": obj_lr,
            "Heuristic_ObjVal": obj_heur,
            "NaiveHeuristic_ObjVal": obj_naive,
            "IP_Time": t_ip,
            "LR_Time": t_lr,
            "Heuristic_Time": t_heur,
            "NaiveHeuristic_Time": t_naive,
            "LR_Gap": gap_lr,
            "Heuristic_Gap": gap_heur,
            "NaiveHeuristic_Gap": gap_naive
        })

        # Control print
        print(f"✅ Scenario {scenario_id:2d} | Instance {instance_id:2d} | "
              f"IP: {obj_ip:,.2f} ({t_ip:.2f}s) | "
              f"LR: {obj_lr:,.2f} ({t_lr:.2f}s) | "
              f"Heuristic: {obj_heur:,.2f} ({t_heur:.2f}s) | "
              f"Naive: {obj_naive:,.2f} ({t_naive:.2f}s)")

    df = pd.DataFrame(results)
    output_path = os.path.join(folder_path, "instance_results_summary.xlsx")
    df.to_excel(output_path, index=False)
    return df

#N, T, J, D, I, I_0, V, V_C, T_lead, C = load_instance_data("base_case/scenario_1_instance_1.xlsx")

#print(solve_ip(N, T, J, D, I, I_0, V, V_C, T_lead, C))
#print(solve_lp_relaxation(N, T, J, D, I, I_0, V, V_C, T_lead, C))
#print(solve_lp_round_fix(N, T, J, D, I, I_0, V, V_C, T_lead, C))

run_experiments("/Users/maximilian/PycharmProjects/OR Midterm Project/generated_structured_20250424_160446")
#run_experiments("/Users/maximilian/PycharmProjects/OR Midterm Project/base_case")