import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

######################## PARAMETERS STARTING HERE ################################
file_path = "OR113-2_midtermProject_data.xlsx"

# Demand and initial inventory
df_demand      = pd.read_excel(file_path, sheet_name="Demand")
df_in_transit  = pd.read_excel(file_path, sheet_name="In-transit")
df_shipping    = pd.read_excel(file_path, sheet_name="Shipping cost")
df_inventory   = pd.read_excel(file_path, sheet_name="Inventory cost")

N = df_demand.shape[0] - 1
T = df_demand.shape[1] - 2
J = df_shipping.shape[1] - 1

# Build data arrays
I0 = np.zeros(N)         # initial on-hand inventory
D  = np.zeros((N, T))    # demand
I  = np.zeros((N, T))    # in-transit arriving this month
for i in range(N):
    I0[i] = df_demand.iloc[i+1,1]
    for t in range(T):
        D[i,t] = df_demand.iloc[i+1, t+2]
for i in range(N):
    for t in range(df_in_transit.shape[1]-1):
        I[i,t] = df_in_transit.iloc[i+1, t+1]

# Cost & volume parameters
C = {
    "H": np.zeros(N),      # holding
    "P": np.zeros(N),      # purchase
    "V": np.zeros((N,J)),  # variable freight
    "F": np.array([100,80,50]),  # fixed freight
    "C": 2750,             # container cost
}
V    = np.zeros(N)         # product volume
for i in range(N):
    C["P"][i] = df_inventory.iloc[i,2]
    C["H"][i] = df_inventory.iloc[i,3]
    V[i]       = df_shipping.iloc[i,3]
    for j in range(J):
        C["V"][i,j] = 0 if j==J-1 else df_shipping.iloc[i,j+1]

# --- NEW: read sales price & compute lost-sales cost ---
S      = np.zeros(N)        # sales price
LS_cost = np.zeros(N)       # lost-sales/unit = S - purchase
for i in range(N):
    S[i]       = df_inventory.iloc[i,1]
    LS_cost[i] = S[i] - C["P"][i]

T_lead = np.array([1,2,3])
V_C    = 30
######################## PARAMETERS ENDING HERE ##################################

# Build the model
model = gp.Model("Inventory_with_LostSales")
model.setParam('MIPGap', 0.0)

S_I = range(N)
S_J = range(J)
S_T = range(T)

# Decision vars
x  = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")   # order qty
v  = model.addVars(S_I, S_T,   vtype=GRB.CONTINUOUS, name="v")    # end-inv
y  = model.addVars(S_J, S_T,   vtype=GRB.BINARY,     name="y")    # method used
z  = model.addVars(S_T,       vtype=GRB.INTEGER,    name="z")    # # containers
ls = model.addVars(S_I, S_T,  vtype=GRB.CONTINUOUS, name="ls")   # lost sales

# Objective: holding + purchasing+freight + container + lost-sales
holding = gp.quicksum(C["H"][i]*v[i,t]
                      for i in S_I for t in S_T)
purch_ship = gp.quicksum((C["P"][i] + C["V"][i,j]) * x[i,j,t]
                        for i in S_I for j in S_J for t in S_T) \
            + gp.quicksum(C["F"][j]*y[j,t]
                          for j in S_J for t in S_T)
container = gp.quicksum(C["C"]*z[t] for t in S_T)
lostsales = gp.quicksum(LS_cost[i]*ls[i,t]
                       for i in S_I for t in S_T)

model.setObjective(holding + purch_ship + container + lostsales, GRB.MINIMIZE)

# Constraints

# Inventory balance with lost sales
for i in S_I:
    for t in S_T:
        # incoming this period
        in_transit = gp.quicksum(x[i,j,t - T_lead[j] + 1]
                                for j in S_J
                                if t - T_lead[j] + 1 >= 0)
        if t == 0:
            model.addConstr(v[i,t] == I0[i] + I[i,t] - D[i,t] + in_transit + ls[i,t],
                            name=f"InvBal_{i}_{t}")
        else:
            model.addConstr(v[i,t] == v[i,t-1] + I[i,t] - D[i,t]
                                          + in_transit + ls[i,t],
                            name=f"InvBal_{i}_{t}")
        # enforce non-negative inventory
        model.addConstr(v[i,t] >= 0, name=f"NonNegInv_{i}_{t}")
        # lost sales >= 0
        model.addConstr(ls[i,t] >= 0, name=f"NonNegLS_{i}_{t}")

# Link orders to shipping methods
M = D.sum()  # big-M
for j in S_J:
    for t in S_T:
        model.addConstr(gp.quicksum(x[i,j,t] for i in S_I) <= M * y[j,t],
                        name=f"ShipMethod_{j}_{t}")

# Container capacity for ocean freight (j=2)
for t in S_T:
    model.addConstr(gp.quicksum(V[i]*x[i,2,t] for i in S_I)
                    <= V_C * z[t],
                    name=f"Container_{t}")

# Nonnegativity & integrality
for i in S_I:
    for j in S_J:
        for t in S_T:
            model.addConstr(x[i,j,t] >= 0, name=f"NonNeg_x_{i}_{j}_{t}")
for j in S_J:
    for t in S_T:
        model.addConstr(y[j,t] >= 0, name=f"Binary_y_{j}_{t}")  # y is binary
for t in S_T:
    model.addConstr(z[t] >= 0, name=f"NonNeg_z_{t}")

# Solve
model.optimize()

# Display
if model.status == GRB.OPTIMAL:
    print("Optimal cost:", model.objVal)
    print("Orders x[i,j,t]>0:")
    for t in S_T:
        for i in S_I:
            for j in S_J:
                if x[i,j,t].x > 1e-6:
                    print(f"  x[{i},{j},{t}] = {x[i,j,t].x:.2f}")
    print("End inv v[i,t]>0:")
    for t in S_T:
        for i in S_I:
            if v[i,t].x > 1e-6:
                print(f"  v[{i},{t}] = {v[i,t].x:.2f}")
    print("Lost sales ls[i,t]>0:")
    for t in S_T:
        for i in S_I:
            if ls[i,t].x > 1e-6:
                print(f"  ls[{i},{t}] = {ls[i,t].x:.2f}")
    print("Containers z[t]>0:")
    for t in S_T:
        if z[t].x > 0:
            print(f"  z[{t}] = {z[t].x:.0f}")
else:
    print("No optimal solution found.")
