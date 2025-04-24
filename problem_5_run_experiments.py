import os
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_and_evaluate_all_instances(folder_path):
    results = []
    current_scenario = None

    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".xlsx"):
            continue

        file_path = os.path.join(folder_path, file)
        xls = pd.ExcelFile(file_path)

        scenario_id = int(file.split("_")[1])
        instance_id = int(file.split("_")[3].split(".")[0])

        # Print when entering a new scenario
        if scenario_id != current_scenario:
            current_scenario = scenario_id
            print(f"🚀 Starting Scenario {scenario_id}...")

        df_demand = pd.read_excel(xls, "Demand")
        df_intransit = pd.read_excel(xls, "In-transit")
        df_shipcost = pd.read_excel(xls, "Shipping cost")
        df_invcost = pd.read_excel(xls, "Inventory cost")
        df_params = pd.read_excel(xls, "Parameters")

        N = df_demand.shape[0]
        T = df_demand.shape[1] - 2
        J = 3
        D = np.zeros((N, T))
        I = np.zeros((N, T))
        I_0 = np.zeros(N)
        V = np.zeros(N)
        C = {"H": np.zeros(N), "P": np.zeros(N), "V": np.zeros((N, J)), "F": np.zeros(J), "C": 0}
        V_C = 30
        T_lead = np.zeros(J)

        for i in range(N):
            I_0[i] = df_demand.iloc[i, 1]
            for t in range(T):
                D[i, t] = df_demand.iloc[i, t + 2]
                I[i, t] = df_intransit.iloc[i, t + 1]

        for i in range(N):
            V[i] = df_shipcost.iloc[i, 4]
            C["V"][i, 0] = df_shipcost.iloc[i, 1]
            C["V"][i, 1] = df_shipcost.iloc[i, 2]
            C["V"][i, 2] = 0

        for i in range(N):
            C["P"][i] = df_invcost.iloc[i, 2]
            C["H"][i] = df_invcost.iloc[i, 3]

        for _, row in df_params.iterrows():
            key = row["Parameter"]
            val = row["Value"]
            if key == "ContainerCost":
                C["C"] = val
            elif key == "ContainerVolume":
                V_C = val
            elif key.startswith("FixedCost_Method"):
                j = int(key[-1]) - 1
                C["F"][j] = val
            elif key.startswith("LeadTime_Method"):
                j = int(key[-1]) - 1
                T_lead[j] = int(val)

        model = gp.Model("LP_Relaxation")
        model.setParam('OutputFlag', 0)

        S_I = range(N)
        S_T = range(T)
        S_J = range(J)

        x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")
        v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")
        y = model.addVars(S_J, S_T, vtype=GRB.CONTINUOUS, name="y")
        z = model.addVars(S_T, vtype=GRB.CONTINUOUS, name="z")

        model.setObjective(
            gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T) +
            gp.quicksum((C["P"][i] + C["V"][i, j]) * x[i, j, t] for i in S_I for j in S_J for t in S_T) +
            gp.quicksum(C["F"][j] * y[j, t] for j in S_J for t in S_T) +
            gp.quicksum(C["C"] * z[t] for t in S_T),
            GRB.MINIMIZE
        )

        for i in S_I:
            for t in S_T:
                in_inventory = gp.quicksum(
                    x[i, j, t - int(T_lead[j]) + 1]
                    for j in S_J if t - int(T_lead[j]) + 1 >= 0
                )
                if t == 0:
                    model.addConstr(v[i, t] == in_inventory + I_0[i] + I[i, t] - D[i, t])
                else:
                    model.addConstr(v[i, t] == v[i, t-1] + in_inventory + I[i, t] - D[i, t])

        M = np.sum(D)
        for j in S_J:
            for t in S_T:
                model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t])

        for t in S_T:
            model.addConstr(gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t])

        t0 = time.time()
        model.optimize()
        t_lp = time.time() - t0
        obj_lp = model.objVal if model.status == GRB.OPTIMAL else None

        # ===== Heuristic (express-only greedy) =====
        t1 = time.time()
        total_cost_heur = 0
        inventory = I_0.copy()

        for t in range(T):
            shipping_cost = 0
            holding_cost = 0
            purchase_cost = 0
            for i in range(N):
                current_inventory = inventory[i] + I[i, t]
                demand = D[i, t]
                order = max(0, demand - current_inventory)
                inventory[i] = current_inventory + order - demand
                shipping_cost += C["V"][i, 0] * order
                purchase_cost += C["P"][i] * order
                holding_cost += C["H"][i] * inventory[i]
            total_cost_heur += shipping_cost + purchase_cost + holding_cost
        t_heur = time.time() - t1
        obj_heur = total_cost_heur

        # ===== Naive Heuristic =====
        t2 = time.time()
        total_cost_naive = 0
        for t in range(T):
            for i in range(N):
                qty = D[i, t]
                total_cost_naive += qty * (C["P"][i] + C["V"][i, 0])
        t_naive = time.time() - t2
        obj_naive = total_cost_naive

        gap_heur = 100 * (obj_heur - obj_lp) / obj_lp
        gap_naive = 100 * (obj_naive - obj_lp) / obj_lp

        results.append({
            "ScenarioID": scenario_id,
            "InstanceID": instance_id,
            "Obj_LP": round(obj_lp, 2),
            "Obj_Heuristic": round(obj_heur, 2),
            "Obj_Naive": round(obj_naive, 2),
            "Time_LP": round(t_lp, 4),
            "Time_Heuristic": round(t_heur, 4),
            "Time_Naive": round(t_naive, 4),
            "Gap_Heuristic": round(gap_heur, 2),
            "Gap_Naive": round(gap_naive, 2)
        })

        if instance_id == 30:
            print(f"✅ Finished testing Scenario {scenario_id} with 30 instances.")

    # Save results to Excel
    df_results = pd.DataFrame(results)
    output_excel = os.path.join(folder_path, "instance_results_summary.xlsx")
    df_results.to_excel(output_excel, index=False)
    print(f"\n🧾 All results saved to: {output_excel}")
    return output_excel

# run experiments
solve_and_evaluate_all_instances("generated_structured_20250424_160446")