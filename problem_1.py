import pandas as pd
import gurobipy as gp
from gurobipy import GRB

FILE    = 'OR113-2_midtermProject_data.xlsx'
Tj      = {1:1, 2:2, 3:3}
CF      = {1:100, 2:80, 3:50}
CC, VC  = 2750, 30.0

xl       = pd.ExcelFile(FILE)

dem_raw  = xl.parse('Demand', header=None)
months   = list(dem_raw.iloc[1,2:8])  # ['March',…,'August']
dem_raw.columns = list(dem_raw.iloc[0,:2]) + months
dem      = (
    dem_raw.drop(index=[0,1])
           .dropna(subset=['Product'])
           .astype({'Product':int,'Initial inventory':float})
           .reset_index(drop=True)
)

it_raw   = xl.parse('In-transit', header=None)
it_cols  = list(it_raw.iloc[1,1:3])
it_raw.columns=['Product']+it_cols
it       = (
    it_raw.drop(index=[0,1])
          .dropna(subset=['Product'])
          .astype(int)
          .reset_index(drop=True)
)

inv      = xl.parse('Inventory cost').set_index('Product')
ship     = xl.parse('Shipping cost').set_index('Product')

periods  = range(1,7)
monthnm  = dict(zip(periods, months))
items    = dem['Product'].tolist()

D  = {
    (r.Product, t): float(r[monthnm[t]])
    for _,r in dem.iterrows() for t in periods
}

I0      = dem.set_index('Product')['Initial inventory'].to_dict()
Iin     = {
    (r.Product,1): float(r['End of March']) for _,r in it.iterrows()
}
Iin.update({
    (r.Product,2): float(r['End of April']) for _,r in it.iterrows()
})

CP      = inv['Purchasing cost'].to_dict()
CH      = inv['Holding cost'].to_dict()
SP      = inv['Sales price'].to_dict()
CLS     = {p: SP[p]-CP[p] for p in items}

CV, V   = {}, {}
for p, row in ship.iterrows():
    CV[(p,1)] = float(row['Express delivery'])
    CV[(p,2)] = float(row['Air freight'])
    CV[(p,3)] = 0.0
    V[p] = float(row['Cubic meter'])


m = gp.Model('OrderingPlan_withLostSales')
m.Params.MIPGap = 0.0


x  = m.addVars(items, Tj.keys(), periods, lb=0, name='x')
y  = m.addVars(Tj.keys(), periods, vtype=GRB.BINARY, name='y')
v  = m.addVars(items, periods, lb=0, name='v')
ls = m.addVars(items, periods, lb=0, name='ls')
z  = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='z')


for i in items:
    for t in periods:
        arrivals = gp.quicksum(
            x[i,j, t-Tj[j]+1]
            for j in Tj if t-Tj[j]+1>=1
        )
        prev_inv = I0[i] if t==1 else v[i,t-1]
        intrans   = Iin.get((i,t), 0.0)

        m.addConstr(
            prev_inv + intrans + arrivals
            - D[(i,t)] == v[i,t] - ls[i,t],
            name=f"bal_{i}_{t}"
        )

        m.addConstr(
            ls[i,t] <= D[(i,t)],
            name=f"lsBound_{i}_{t}"
        )

for j in Tj:
    lt = Tj[j]
    for t in periods:
        rem = sum(
            D[(i,τ)]
            for i in items
            for τ in range(t+lt-1, 7)
            if τ in periods
        )
        m.addConstr(
            gp.quicksum(x[i,j,t] for i in items)
            <= rem * y[j,t],
            name=f"bigM_{j}_{t}"
        )


for t in periods:
    m.addConstr(
        gp.quicksum(V[i]*x[i,3,t] for i in items)
        <= VC * z[t],
        name=f"capSea_{t}"
    )


holding   = gp.quicksum(CH[i]*v[i,t]            for i in items for t in periods)
purchase  = gp.quicksum(CP[i]*x[i,j,t]          for i in items for j in Tj for t in periods)
varship   = gp.quicksum(CV[(i,j)]*x[i,j,t]      for i in items for j in Tj for t in periods)
fixship   = gp.quicksum(CF[j]*y[j,t]            for j in Tj for t in periods)
contcost  = gp.quicksum(CC*z[t]                 for t in periods)
lostsales  = gp.quicksum(CLS[i]*ls[i,t]         for i in items for t in periods)

m.setObjective(
    holding + purchase + varship + fixship + contcost + lostsales,
    GRB.MINIMIZE
)

m.Params.OutputFlag = 1
m.optimize()

import pandas as pd

x_sol = m.getAttr('X', x)

order_plan = []
for (i, j, t), val in x_sol.items():
    if val > 1e-3:
        order_plan.append({
            "Period": monthnm[t],
            "Product": i,
            "Shipping Method": j,
            "Quantity": round(val, 2)
        })

df = pd.DataFrame(order_plan)

month_order = ["March", "April", "May", "June", "July", "August"]

df["Period"] = pd.Categorical(df["Period"], categories=month_order, ordered=True)

df.sort_values(by=["Period", "Product", "Shipping Method"], inplace=True)

print(df)