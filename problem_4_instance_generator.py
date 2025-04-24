import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

# 🔁 Reproducibility
np.random.seed(42)
random.seed(42)

# 📁 Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./generated_structured_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 🧩 Scenario setup: scale, container cost, holding %
scenarios = {
    1: {"scale": "medium", "container_cost": 2750, "holding_pct": 0.02},
    2: {"scale": "small", "container_cost": 2750, "holding_pct": 0.02},
    3: {"scale": "large", "container_cost": 2750, "holding_pct": 0.02},
    4: {"scale": "medium", "container_cost": 1375, "holding_pct": 0.02},
    5: {"scale": "medium", "container_cost": 5500, "holding_pct": 0.02},
    6: {"scale": "medium", "container_cost": 2750, "holding_pct": 0.01},
    7: {"scale": "medium", "container_cost": 2750, "holding_pct": 0.04},
}

scale_settings = {
    "small": (10, 6),
    "medium": (100, 20),
    "large": (500, 50),
}

# 🚛 Fixed global parameters
fixed_cf = {1: 100, 2: 80, 3: 50}
fixed_tj = {1: 1, 2: 2, 3: 3}
container_volume = 30
max_lead_time = max(fixed_tj.values())  # Max time a shipment can be in transit

# 🔄 Loop over scenarios and generate 30 instances each
for scenario_id, settings in scenarios.items():
    num_products, num_periods = scale_settings[settings["scale"]]
    holding_pct = settings["holding_pct"]
    container_cost = settings["container_cost"]

    for instance_id in range(1, 31):
        demand_rows = []
        intransit_rows = []
        shipcost_rows = []
        invcost_rows = []

        for i in range(1, num_products + 1):
            # Randomized parameters per Table 13
            purchase_cost = np.random.randint(1000, 10001)
            holding_cost = round(purchase_cost * holding_pct, 2)

            cv1 = round(np.random.uniform(40, 100), 2)
            alpha = np.random.uniform(0.4, 0.6)
            cv2 = round(alpha * cv1, 2)
            cv3 = 0  # Ocean shipping cost is handled via container logic

            volume = round(np.random.uniform(0.01, 1.0), 4)

            # Demand and inventory
            demands = [np.random.randint(0, 201) for _ in range(num_periods)]
            init_inventory = np.random.randint(demands[0], 401)

            # Only periods 1–3 may have in-transit inventory
            in_transit = []
            for t in range(1, num_periods + 1):
                if t <= max_lead_time:
                    val = 0 if np.random.rand() < 0.5 else np.random.randint(0, 51)
                else:
                    val = 0
                in_transit.append(val)

            # Append all product-level data
            demand_rows.append([i, init_inventory] + demands)
            intransit_rows.append([i] + in_transit)
            shipcost_rows.append([i, cv1, cv2, cv3, volume])
            invcost_rows.append([i, f"Product_{i}", purchase_cost, holding_cost])

        # 💾 Convert to DataFrames
        df_demand = pd.DataFrame(demand_rows, columns=["ProductID", "InitInventory"] + [f"Period {t+1}" for t in range(num_periods)])
        df_intransit = pd.DataFrame(intransit_rows, columns=["ProductID"] + [f"Period {t+1}" for t in range(num_periods)])
        df_shipcost = pd.DataFrame(shipcost_rows, columns=["ProductID", "CV1", "CV2", "CV3", "Volume"])
        df_invcost = pd.DataFrame(invcost_rows, columns=["ProductID", "Name", "PurchaseCost", "HoldingCost"])

        # 📄 Parameter sheet (adds scale info too)
        param_data = {
            "Parameter": ["NumProducts", "NumPeriods", "ContainerCost", "ContainerVolume", "HoldingPct"] +
                         [f"FixedCost_Method{j}" for j in fixed_cf] +
                         [f"LeadTime_Method{j}" for j in fixed_tj],
            "Value": [num_products, num_periods, container_cost, container_volume, holding_pct] +
                     [fixed_cf[j] for j in fixed_cf] +
                     [fixed_tj[j] for j in fixed_tj]
        }
        df_params = pd.DataFrame(param_data)

        # 🔄 Save all sheets to one Excel file
        file_name = f"scenario_{scenario_id}_instance_{instance_id}.xlsx"
        file_path = os.path.join(output_dir, file_name)
        with pd.ExcelWriter(file_path) as writer:
            df_demand.to_excel(writer, sheet_name="Demand", index=False)
            df_intransit.to_excel(writer, sheet_name="In-transit", index=False)
            df_shipcost.to_excel(writer, sheet_name="Shipping cost", index=False)
            df_invcost.to_excel(writer, sheet_name="Inventory cost", index=False)
            df_params.to_excel(writer, sheet_name="Parameters", index=False)

    print(f"✅ Finished generating all 30 instances for Scenario {scenario_id}")

print(f"\n📂 All files saved in folder: {output_dir}")