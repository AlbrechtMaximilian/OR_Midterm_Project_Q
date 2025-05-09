import os
import time
from time import perf_counter
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

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

        S_I = range(N)
        S_T = range(T)
        S_J = range(J)
        M = np.sum(D)

        # ===== LP Relaxation =====
        t_lp = perf_counter()
        model = gp.Model("LP_Relaxation")
        model.setParam('OutputFlag', 0)

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
                    model.addConstr(v[i, t] == v[i, t - 1] + in_inventory + I[i, t] - D[i, t])

        for j in S_J:
            for t in S_T:
                model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t])

        for t in S_T:
            model.addConstr(gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t])

        model.optimize()
        obj_lp = model.objVal if model.status == GRB.OPTIMAL else None
        t_lp = perf_counter() - t_lp

        # ===== IP Model =====
        t_ip = perf_counter()
        model_ip = gp.Model("IP_Model")
        model_ip.setParam('OutputFlag', 0)
        model_ip.setParam("TimeLimit", 60)

        x_ip = model_ip.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x_ip")
        v_ip = model_ip.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v_ip")
        y_ip = model_ip.addVars(S_J, S_T, vtype=GRB.BINARY, name="y_ip")
        z_ip = model_ip.addVars(S_T, vtype=GRB.INTEGER, name="z_ip")

        model_ip.setObjective(
            gp.quicksum(C["H"][i] * v_ip[i, t] for i in S_I for t in S_T) +
            gp.quicksum((C["P"][i] + C["V"][i, j]) * x_ip[i, j, t] for i in S_I for j in S_J for t in S_T) +
            gp.quicksum(C["F"][j] * y_ip[j, t] for j in S_J for t in S_T) +
            gp.quicksum(C["C"] * z_ip[t] for t in S_T),
            GRB.MINIMIZE
        )

        for i in S_I:
            for t in S_T:
                in_inventory_ip = gp.quicksum(
                    x_ip[i, j, t - int(T_lead[j]) + 1]
                    for j in S_J if t - int(T_lead[j]) + 1 >= 0
                )
                if t == 0:
                    model_ip.addConstr(v_ip[i, t] == in_inventory_ip + I_0[i] + I[i, t] - D[i, t])
                else:
                    model_ip.addConstr(v_ip[i, t] == v_ip[i, t - 1] + in_inventory_ip + I[i, t] - D[i, t])
                    model_ip.addConstr(v_ip[i, t - 1] >= D[i, t])

        for j in S_J:
            for t in S_T:
                model_ip.addConstr(gp.quicksum(x_ip[i, j, t] for i in S_I) <= M * y_ip[j, t])

        for t in S_T:
            model_ip.addConstr(gp.quicksum(V[i] * x_ip[i, 2, t] for i in S_I) <= V_C * z_ip[t])


        model_ip.optimize()

        obj_ip = model_ip.objVal if model_ip.status == GRB.OPTIMAL else None
        if obj_ip is None:
            print(f"⚠️  IP not solved optimally for Scenario {scenario_id}, Instance {instance_id}.")
        t_ip = perf_counter() - t_ip

        # ===== Heuristic (LP-based round-and-fix heuristic) =====
        t1 = perf_counter()  # Start total heuristic timing

        # Step 1: Solve LP relaxation
        model = gp.Model("Heuristic_LP_Relaxation")
        model.setParam('OutputFlag', 0)

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
                    model.addConstr(v[i, t] == v[i, t - 1] + in_inventory + I[i, t] - D[i, t])

        for j in S_J:
            for t in S_T:
                model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t])

        for t in S_T:
            model.addConstr(gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t])

        model.optimize()

        # Step 2: Round y and z (based on LP solution)
        y_rounded = {(j, t): 1 if y[j, t].X > 0 else 0 for j in S_J for t in S_T}
        z_rounded = {t: math.ceil(z[t].X) if z[t].X > 0 else 0 for t in S_T}

        # Step 3: Re-solve with fixed y and z
        model_fix = gp.Model("Heuristic_Fix")
        model_fix.setParam("OutputFlag", 0)

        x_f = model_fix.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x_f")
        v_f = model_fix.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v_f")

        model_fix.setObjective(
            gp.quicksum(C["H"][i] * v_f[i, t] for i in S_I for t in S_T) +
            gp.quicksum((C["P"][i] + C["V"][i, j]) * x_f[i, j, t] for i in S_I for j in S_J for t in S_T) +
            gp.quicksum(C["F"][j] * y_rounded[j, t] for j in S_J for t in S_T) +
            gp.quicksum(C["C"] * z_rounded[t] for t in S_T),
            GRB.MINIMIZE
        )

        for i in S_I:
            for t in S_T:
                in_flow = gp.quicksum(x_f[i, j, t - int(T_lead[j]) + 1]
                                      for j in S_J if t - int(T_lead[j]) + 1 >= 0)
                if t == 0:
                    model_fix.addConstr(v_f[i, t] == I_0[i] + I[i, t] + in_flow - D[i, t])
                else:
                    model_fix.addConstr(v_f[i, t] == v_f[i, t - 1] + I[i, t] + in_flow - D[i, t])

        for j in S_J:
            for t in S_T:
                model_fix.addConstr(gp.quicksum(x_f[i, j, t] for i in S_I) <= M * y_rounded[j, t])

        for t in S_T:
            model_fix.addConstr(gp.quicksum(V[i] * x_f[i, 2, t] for i in S_I) <= V_C * z_rounded[t])

        model_fix.optimize()
        obj_heur = model_fix.objVal if model_fix.status == GRB.OPTIMAL else None
        t_heur = perf_counter() - t1

        """"# ===== Heuristic (direct MIP with relaxed structure, no rounding) =====
        t1 = perf_counter()

        model_heur = gp.Model("Heuristic_MIP")
        model_heur.setParam("OutputFlag", 0)
        model_heur.setParam("MIPGap", 0.0)

        x_h = model_heur.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")
        v_h = model_heur.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")
        y_h = model_heur.addVars(S_J, S_T, vtype=GRB.BINARY, name="y")
        z_h = model_heur.addVars(S_T, vtype=GRB.INTEGER, name="z")

        model_heur.setObjective(
            gp.quicksum(C["H"][i]*v_h[i,t] for i in S_I for t in S_T)
          + gp.quicksum((C["P"][i]+C["V"][i,j])*x_h[i,j,t] for i in S_I for j in S_J for t in S_T)
          + gp.quicksum(C["F"][j]*y_h[j,t] for j in S_J for t in S_T)
          + gp.quicksum(C["C"]*z_h[t] for t in S_T),
            GRB.MINIMIZE
        )

        for i in S_I:
            for t in S_T:
                arrive = gp.LinExpr()
                for j in S_J:
                    tau = t - (int(T_lead[j])-1)
                    if tau >= 0:
                        arrive += x_h[i,j,tau]
                if t == 0:
                    model_heur.addConstr(v_h[i,0] == I_0[i] + I[i,0] + arrive - D[i,0])
                else:
                    model_heur.addConstr(v_h[i,t] == v_h[i,t-1] + I[i,t] + arrive - D[i,t])
                    model_heur.addConstr(v_h[i,t-1] >= D[i,t])  # No stockout

        for j in S_J:
            for t in S_T:
                model_heur.addConstr(
                    gp.quicksum(x_h[i,j,t] for i in S_I) <= M * y_h[j,t]
                )

        for t in S_T:
            model_heur.addConstr(
                gp.quicksum(V[i]*x_h[i,2,t] for i in S_I) <= V_C * z_h[t]
            )

        model_heur.optimize()
        t_heur = perf_counter() - t1
        obj_heur = model_heur.objVal if model_heur.status == GRB.OPTIMAL else None"""

        """"# ===== Heuristic (Express-only greedy) =====
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
        obj_heur = total_cost_heur"""

        # ===== Naive Heuristic =====
        t2 = perf_counter()
        total_cost_naive = 0
        for t in range(T):
            for i in range(N):
                qty = D[i, t]
                total_cost_naive += qty * (C["P"][i] + C["V"][i, 0])
        t_naive = perf_counter() - t2
        obj_naive = total_cost_naive

        """t2 = perf_counter()
        total_cost_naive = 0
        
        for t in range(T):
            for i in range(N):
                qty = D[i, t]
                # Produktions- und variable Versandkosten
                total_cost_naive += qty * (C["P"][i] + C["V"][i, 0])
                # fixe Express-Kosten pro Auftrag (hier: pro Produkt und Periode)
                total_cost_naive += CF[0]
        
        t_naive = perf_counter() - t2
        obj_naive = total_cost_naive"""

        if obj_ip is not None and model_ip.status == gp.GRB.OPTIMAL:
            gap_lp = 100 * (obj_lp - obj_ip) / obj_ip
            gap_heur = 100 * (obj_heur - obj_ip) / obj_ip
            gap_naive = 100 * (obj_naive - obj_ip) / obj_ip
        else:
            # fallback: use LP as proxy if IP not optimal
            print("error")
            gap_lp = None
            gap_heur = 100 * (obj_heur - obj_lp) / obj_lp
            gap_naive = 100 * (obj_naive - obj_lp) / obj_lp

        results.append({
            "ScenarioID": scenario_id,
            "InstanceID": instance_id,
            "Obj_IP": obj_ip,
            "Obj_LP": obj_lp,
            "Obj_Heuristic": obj_heur,
            "Obj_Naive": obj_naive,
            "Time_IP": t_ip,
            "Time_LP": t_lp,
            "Time_Heuristic": t_heur,
            "Time_Naive": t_naive,
            "Gap_Heuristic": gap_heur,
            "Gap_Naive": gap_naive,
            "Gap_LP": gap_lp
        })

        if instance_id == 30:
            print(f"✅ Finished testing Scenario {scenario_id} with 30 instances.")

    df_results = pd.DataFrame(results)
    df_results.sort_values(by=["ScenarioID", "InstanceID"], inplace=True)  # 🔥 sort properly
    output_excel = os.path.join(folder_path, "results.xlsx")
    df_results.to_excel(output_excel, index=False)
    print(f"\n🧾 All results saved to: {output_excel}")
    return output_excel

# run experiments
solve_and_evaluate_all_instances("generated_structured_20250424_160446")