import pandas as pd
import numpy as np

######################## PARAMETERS STARTING HERE ################################
# Read the Excel file from the 'Demand' sheet
file_path = "OR113-2_midtermProject_data.xlsx"
df_demand = pd.read_excel(file_path, sheet_name="Demand")
N = df_demand.shape[0] - 1
T = df_demand.shape[1] - 2
print("N:", N, "T:", T)

I = np.zeros([N, T])
D = np.zeros([N, T])
I_0 = np.zeros([N])

for i in range(N):
    I_0[i] = df_demand.iloc[i+1, 1]
    for t in range(T):
        D[i, t] = df_demand.iloc[i+1, t+2]

df_in_transit = pd.read_excel(file_path, sheet_name="In-transit")
for i in range(N):
    for t in range(df_in_transit.shape[1] - 1):
        I[i, t] = df_in_transit.iloc[i+1, t+1]

df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
J = df_shipping_cost.shape[1] - 1
df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")

C = {
    "H": np.zeros([N]),
    "P": np.zeros([N]),
    "V": np.zeros([N, J]),
    "F": np.array([100, 80, 50]),
    "C": 2750,
}
V = np.zeros([N])
V_C = 30
for i in range(N):
    C["H"][i] = df_inventory_cost.iloc[i, 3]
    C["P"][i] = df_inventory_cost.iloc[i, 2]
    V[i] = df_shipping_cost.iloc[i, 3]
    for j in range(J):
        if j == J - 1:
            C["V"][i, j] = 0
        else:
            C["V"][i, j] = df_shipping_cost.iloc[i, j+1]

S       = np.zeros(N)
LS_cost = np.zeros(N)
for i in range(N):
    S[i]       = df_inventory_cost.iloc[i, 1]
    LS_cost[i] = S[i] - C["P"][i]

BO_cost = np.array([535, 250, 1345, 980, 345, 1575, 705, 820, 1650, 930])
P_frac  = np.array([0.0, 0.7, 0.1, 1.0, 1.0, 0.3, 0.6, 0.2, 0.1, 0.5])

T_lead = np.array([1, 2, 3])
######################## PARAMETERS ENDING HERE ##################################

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("InventoryManagement")
model.setParam('MIPGap', 0.0)

S_I = range(N)
S_T = range(T)
S_J = range(J)

x  = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")
v  = model.addVars(S_I, S_T,   vtype=GRB.CONTINUOUS, name="v")
y  = model.addVars(S_J, S_T,   vtype=GRB.BINARY,     name="y")
z  = model.addVars(S_T,       vtype=GRB.INTEGER,    name="z")
s  = model.addVars(S_I, S_T,   vtype=GRB.CONTINUOUS, name="s")
bo = model.addVars(S_I, S_T,   vtype=GRB.CONTINUOUS, name="bo")
ls = model.addVars(S_I, S_T,   vtype=GRB.CONTINUOUS, name="ls")

# Objective: holding + purch+ship + container + backorder + lost‐sales
holding = gp.quicksum(C["H"][i]*v[i,t] for i in S_I for t in S_T)
purch_ship = (
    gp.quicksum((C["P"][i] + C["V"][i,j]) * x[i,j,t]
                for i in S_I for j in S_J for t in S_T)
  + gp.quicksum(C["F"][j] * y[j,t] for j in S_J for t in S_T)
)
container   = gp.quicksum(C["C"] * z[t] for t in S_T)
backorder   = gp.quicksum(BO_cost[i] * bo[i,t] for i in S_I for t in S_T)
lostsales   = gp.quicksum(LS_cost[i] * ls[i,t] for i in S_I for t in S_T)

model.setObjective(holding + purch_ship + container + backorder + lostsales, GRB.MINIMIZE)

# Constraints
J_in_inventory = np.array([1, 2, 3, 3, 3, 3])

for i in S_I:
    for t in S_T:
        in_inventory = gp.quicksum(
            x[i,j,t - T_lead[j] + 1]
            for j in S_J
            if t - T_lead[j] + 1 >= 0
        )
        if t == 0:
            model.addConstr(
                v[i,t] == I_0[i] + I[i,t] + in_inventory
                         - D[i,t]
                         - 0
                         + s[i,t],
                name=f"InvBalance_{i}_{t}"
            )
        else:
            model.addConstr(
                v[i,t] == v[i,t-1]
                         + I[i,t]
                         + in_inventory
                         - D[i,t]
                         - bo[i,t-1]
                         + s[i,t],
                name=f"InvBalance_{i}_{t}"
            )
        model.addConstr(v[i,t] >= 0, name=f"NonNeg_v_{i}_{t}")
        model.addConstr(s[i,t]   >= 0, name=f"NonNeg_s_{i}_{t}")
        model.addConstr(bo[i,t] == P_frac[i] * s[i,t],      name=f"BackSplit_{i}_{t}")
        model.addConstr(ls[i,t] == (1 - P_frac[i]) * s[i,t],name=f"LostSplit_{i}_{t}")

M = D.sum()
for j in S_J:
    for t in S_T:
        model.addConstr(
            gp.quicksum(x[i,j,t] for i in S_I) <= M * y[j,t],
            name=f"ShippingMethod_{j}_{t}"
        )

for t in S_T:
    model.addConstr(
        gp.quicksum(V[i] * x[i,2,t] for i in S_I) <= V_C * z[t],
        name=f"Container_{t}"
    )
for i in S_I:
    for t in S_T:
        model.addConstr(bo[i,t] >= 0, name=f"NonNeg_bo_{i}_{t}")
        model.addConstr(ls[i,t] >= 0, name=f"NonNeg_ls_{i}_{t}")

model.optimize()