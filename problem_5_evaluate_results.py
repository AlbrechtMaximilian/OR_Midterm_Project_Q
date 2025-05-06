import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results_from_excel(result_excel_path):
    # Load data
    df = pd.read_excel(result_excel_path)
    output_dir = "/Users/maximilian/PycharmProjects/OR Midterm Project/plots"
    os.makedirs(output_dir, exist_ok=True)  # ensure directory exists
    sns.set(style="whitegrid")

    # 1. 📈 Objective Value Comparison (log scale, including IP)
    plt.figure(figsize=(12, 6))
    df_obj = df.melt(
        id_vars=["ScenarioID", "InstanceID"],
        value_vars=["Obj_IP", "Obj_LP", "Obj_Heuristic", "Obj_Naive"],
        var_name="Method", value_name="Objective Value"
    )
    sns.boxplot(data=df_obj, x="ScenarioID", y="Objective Value", hue="Method", palette="Set2")
    plt.yscale("log")
    plt.title("Objective Value Comparison by Scenario (Log Scale)")
    plt.xlabel("Scenario ID")
    plt.ylabel("Objective Value (log scale)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "objective_values.png"), dpi=300)
    plt.show()

    # 2. ⏱ Computation Time per Scenario (log scale, including IP)
    plt.figure(figsize=(12, 6))
    df_time = df.melt(
        id_vars=["ScenarioID"],
        value_vars=["Time_IP", "Time_LP", "Time_Heuristic", "Time_Naive"],
        var_name="Method", value_name="Time (s)"
    )
    sns.boxplot(data=df_time, x="ScenarioID", y="Time (s)", hue="Method", palette="Set3")
    plt.yscale("log")
    plt.title("Computation Time per Scenario (Log Scale)")
    plt.ylabel("Time (log scale, seconds)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "computation_times.png"), dpi=300)
    plt.show()

    # 3. 📦 Optimality Gaps to IP
    plt.figure(figsize=(12, 6))

    # Use existing gap columns if they exist, otherwise compute
    if all(col in df.columns for col in ["Gap_Heuristic", "Gap_Naive", "Gap_LP"]):
        df_gap = df.melt(
            id_vars=["ScenarioID"],
            value_vars=["Gap_Heuristic", "Gap_Naive", "Gap_LP"],
            var_name="Method", value_name="Gap (%)"
        )
        df_gap["Method"] = df_gap["Method"].map({
            "Gap_Heuristic": "Heuristic vs IP",
            "Gap_Naive": "Naive vs IP",
            "Gap_LP": "LP vs IP"
        })
    else:
        df = df[df["Obj_IP"].notnull()].copy()
        df["Gap_Heuristic_toIP"] = 100 * (df["Obj_Heuristic"] - df["Obj_IP"]) / df["Obj_IP"]
        df["Gap_Naive_toIP"] = 100 * (df["Obj_Naive"] - df["Obj_IP"]) / df["Obj_IP"]
        df["Gap_LP_toIP"] = 100 * (df["Obj_LP"] - df["Obj_IP"]) / df["Obj_IP"]
        df_gap = df.melt(
            id_vars=["ScenarioID"],
            value_vars=["Gap_Heuristic_toIP", "Gap_Naive_toIP", "Gap_LP_toIP"],
            var_name="Method", value_name="Gap (%)"
        )
        df_gap["Method"] = df_gap["Method"].map({
            "Gap_Heuristic_toIP": "Heuristic vs IP",
            "Gap_Naive_toIP": "Naive vs IP",
            "Gap_LP_toIP": "LP vs IP"
        })

    sns.boxplot(data=df_gap, x="ScenarioID", y="Gap (%)", hue="Method",
                palette=["skyblue", "salmon", "gray"])
    plt.title("Optimality Gaps to IP per Scenario")
    plt.ylabel("Gap (%)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimality_gaps_to_ip.png"), dpi=300)
    plt.show()

# 🔧 Run it
plot_results_from_excel("generated_structured_20250424_160446/instance_results_summary.xlsx")