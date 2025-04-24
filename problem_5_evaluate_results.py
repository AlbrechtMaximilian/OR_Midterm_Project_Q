import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results_from_excel(result_excel_path):
    # Load data
    df = pd.read_excel(result_excel_path)

    # Global Seaborn style
    sns.set(style="whitegrid")

    # 1. 📈 Objective Value Comparison (per scenario, log scale)
    plt.figure(figsize=(12, 6))
    df_obj = df.melt(
        id_vars=["ScenarioID", "InstanceID"],
        value_vars=["Obj_LP", "Obj_Heuristic", "Obj_Naive"],
        var_name="Method", value_name="Objective Value"
    )
    sns.boxplot(data=df_obj, x="ScenarioID", y="Objective Value", hue="Method", palette="Set2")
    plt.yscale("log")
    plt.title("Objective Value Comparison by Scenario (Log Scale)")
    plt.xlabel("Scenario ID")
    plt.ylabel("Objective Value (log scale)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.show()

    # 2. ⏱ Computation Time per Scenario (side-by-side)
    plt.figure(figsize=(12, 6))
    df_time_scenarios = df.melt(
        id_vars=["ScenarioID"],
        value_vars=["Time_LP", "Time_Heuristic", "Time_Naive"],
        var_name="Method",
        value_name="Time (s)"
    )
    sns.boxplot(data=df_time_scenarios, x="ScenarioID", y="Time (s)", hue="Method", palette="Set3")
    plt.title("Computation Time per Scenario")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.show()

    # 3. 📦 Optimality Gaps by Scenario (side-by-side)
    plt.figure(figsize=(12, 6))
    df_gap_scenarios = df.melt(
        id_vars=["ScenarioID"],
        value_vars=["Gap_Heuristic", "Gap_Naive"],
        var_name="Method",
        value_name="Gap (%)"
    )
    sns.boxplot(data=df_gap_scenarios, x="ScenarioID", y="Gap (%)", hue="Method", palette=["skyblue", "salmon"])
    plt.title("Optimality Gaps per Scenario")
    plt.ylabel("Gap (%)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.show()

# run eevaluation
plot_results_from_excel("generated_structured_20250424_160446/instance_results_summary.xlsx")