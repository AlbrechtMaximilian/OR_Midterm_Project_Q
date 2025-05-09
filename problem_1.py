# -*- coding: utf-8 -*-
"""
===============================================================================
  OR113‑2  ·  Mid‑Term Project – Problem 1 (Lost‑Sales Model)
  -----------------------------------------------------------------------------
  • Reads  :  OR113‑2_midtermProject_data.xlsx
  • Solves :  Mixed‑Integer LP with lost‑sales, lead times & container capacity
  • Writes :  OptimalPlan.csv   – every non‑zero order line
              CostBreakdown.csv – cost components & total
===============================================================================
"""

# ------------------------------------------------------------------ 0 IMPORTS
import pandas as pd
import numpy  as np
import gurobipy as gp
from   gurobipy import GRB

# ------------------------------------------------------------------ 1 INPUT
FILE = "OR113-2_midtermProject_data.xlsx"             # <‑‑ path to data file

## 1.1 Demand & initial inventory
demand_raw = pd.read_excel(FILE, sheet_name="Demand")
months     = list(demand_raw.columns[2:])              # ['Mar', … 'Aug']
N          = len(demand_raw) - 1                       # #products
T          = len(months)                               # 6 months
items, periods = range(N), range(T)

I0 = demand_raw["Initial inventory"].iloc[1:].to_numpy()      # (N,)
D  = demand_raw.iloc[1:, 2:].to_numpy()                       # (N×T)

## 1.2 In‑transit inventory already on the way
it_raw  = pd.read_excel(FILE, sheet_name="In-transit")
Itrans  = np.zeros((N, T))
for t, col in enumerate(it_raw.columns[1:]):                 # “End of …”
    if t < T:
        Itrans[:, t] = it_raw[col].iloc[1:]

## 1.3 Cost & shipping parameters
inv  = pd.read_excel(FILE, sheet_name="Inventory cost")
ship = pd.read_excel(FILE, sheet_name="Shipping cost")

H      = inv["Holding cost"   ].to_numpy()            # €/unit‑month
P      = inv["Purchasing cost"].to_numpy()            # €/unit
Sales  = inv["Sales price"    ].to_numpy()            # €/unit
CLS    = Sales - P                                    # lost‑sales penalty €/unit

J      = 3                                            # 0=Express,1=Air,2=Sea
Vcost  = np.zeros((N, J))                             # variable shipping €/unit
for i in items:
    Vcost[i, 0] = ship.iloc[i, 1]                     # express
    Vcost[i, 1] = ship.iloc[i, 2]                     # air
    # sea variable cost = 0 (container fee covers it)

FixedShip     = np.array([100, 80, 50])               # € per mode/month
Lead          = np.array([1, 2, 3])                   # lead times [months]
VolItem       = ship["Cubic meter"].to_numpy()        # m³ per unit
ContainerFee  = 2750                                  # € per sea container
ContainerCap  = 30.0                                  # m³ capacity

# ------------------------------------------------------------------ 2 MODEL
M = D.sum()                                           # big‑M for linking
m = gp.Model("LostSalesOrdering")
m.Params.OutputFlag = 1
m.Params.MIPGap     = 0.0

## 2.1 Decision variables
x  = m.addVars(N, J, T, lb=0,          name="x")          # order qty
v  = m.addVars(N,    T, lb=0,          name="v")          # ending inventory
ls = m.addVars(N,    T, lb=0,          name="ls")         # lost sales
y  = m.addVars(J,    T, vtype=GRB.BINARY, name="y")       # mode active?
z  = m.addVars(      T, vtype=GRB.INTEGER, lb=0, name="z")# sea containers

## 2.2 Inventory balance & lost‑sales limit
for i in items:
    for t in periods:
        arrivals = gp.quicksum(                           # units arriving in t
            x[i, j, t - Lead[j] + 1]
            for j in range(J) if t - Lead[j] + 1 >= 0
        )
        lhs = (I0[i] if t == 0 else v[i, t-1]) + Itrans[i, t] + arrivals
        m.addConstr(lhs - D[i, t] == v[i, t] - ls[i, t],  f"bal_{i}_{t}")
        m.addConstr(ls[i, t] <= D[i, t],                  f"capLS_{i}_{t}")

## 2.3 Link: no orders in mode j at month t unless y[j,t] = 1
for j in range(J):
    for t in periods:
        m.addConstr(gp.quicksum(x[i, j, t] for i in items) <= M * y[j, t],
                    f"link_{j}_{t}")

## 2.4 Container capacity for sea shipments (mode 2)
for t in periods:
    m.addConstr(gp.quicksum(VolItem[i]*x[i, 2, t] for i in items)
                <= ContainerCap * z[t],                     f"capSea_{t}")

## 2.5 Objective – total cost minimisation
cost = (
      gp.quicksum(H[i]*v[i,t]                 for i in items for t in periods)
    + gp.quicksum((P[i]+Vcost[i,j])*x[i,j,t]  for i in items for j in range(J) for t in periods)
    + gp.quicksum(FixedShip[j]*y[j,t]         for j in range(J) for t in periods)
    + gp.quicksum(ContainerFee*z[t]           for t in periods)
    + gp.quicksum(CLS[i]*ls[i,t]              for i in items for t in periods)
)
m.setObjective(cost, GRB.MINIMIZE)

# ------------------------------------------------------------------ 3 SOLVE
m.optimize()

# ------------------------------------------------------------------ 4 EXPORT
if m.status == GRB.OPTIMAL:

    # 4.1 Order plan
    plan = []
    mode_name = {0: "Express", 1: "Air", 2: "Sea"}
    for t in periods:
        for j in range(J):
            for i in items:
                qty = x[i, j, t].X
                if qty > 1e-6:                               # filter zeros
                    plan.append([months[t], mode_name[j], i+1, round(qty, 2)])
    pd.DataFrame(plan, columns=["Month", "Mode", "Product", "Quantity"]) \
        .to_csv("OptimalPlan.csv", index=False)

    # 4.2 Cost breakdown
    breakdown = [
        ["Holding",   sum(H[i]*v[i,t].X                 for i in items for t in periods)],
        ["Purchasing",sum(P[i]*x[i,j,t].X               for i in items for j in range(J) for t in periods)],
        ["Var ship",  sum(Vcost[i,j]*x[i,j,t].X         for i in items for j in range(J) for t in periods)],
        ["Fixed ship",sum(FixedShip[j]*y[j,t].X         for j in range(J) for t in periods)],
        ["Container", sum(ContainerFee*z[t].X           for t in periods)],
        ["Lost sales",sum(CLS[i]*ls[i,t].X              for i in items for t in periods)],
        ["TOTAL",     m.ObjVal]
    ]
    pd.DataFrame(breakdown, columns=["Cost component", "Amount"]) \
        .to_csv("CostBreakdown.csv", index=False)

    print("\n✅ OptimalPlan.csv and CostBreakdown.csv written.\n")

else:
    print("❌ No optimal solution found.")