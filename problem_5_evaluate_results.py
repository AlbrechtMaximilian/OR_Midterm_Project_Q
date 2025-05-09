import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results_from_excel(result_excel_path):
    # Load data
    df = pd.read_excel(result_excel_path)
    output_dir = "/Users/maximilian/PycharmProjects/OR Midterm Project/plots"
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # ✅ Updated color palette
    color_palette = {
        "IP": "blue",
        "LP Relaxation": "green",
        "Heuristic": "yellow",
        "Naive": "red",
        "Heuristic vs IP": "yellow",
        "Naive vs IP": "red",
        "LP vs IP": "green"
    }

    # 1. 📈 Objective Value Comparison
    plt.figure(figsize=(12, 6))
    df_obj = df.melt(
        id_vars=["ScenarioID", "InstanceID"],
        value_vars=["IP_ObjVal", "LR_ObjVal", "Heuristic_ObjVal", "NaiveHeuristic_ObjVal"],
        var_name="Method", value_name="Objective Value"
    )
    df_obj["Method"] = df_obj["Method"].replace({
        "IP_ObjVal": "IP",
        "LR_ObjVal": "LP Relaxation",
        "Heuristic_ObjVal": "Heuristic",
        "NaiveHeuristic_ObjVal": "Naive"
    })
    sns.boxplot(
        data=df_obj, x="ScenarioID", y="Objective Value", hue="Method",
        palette={key: color_palette[key] for key in df_obj["Method"].unique()}
    )
    plt.yscale("log")
    plt.title("Objective Value Comparison by Scenario (Log Scale)")
    plt.xlabel("Scenario ID")
    plt.ylabel("Objective Value (log scale)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "objective_values.png"), dpi=300)
    plt.show()

    # 2. ⏱ Computation Time
    plt.figure(figsize=(12, 6))
    df_time = df.melt(
        id_vars=["ScenarioID"],
        value_vars=["IP_Time", "LR_Time", "Heuristic_Time", "NaiveHeuristic_Time"],
        var_name="Method", value_name="Time (s)"
    )
    df_time["Method"] = df_time["Method"].replace({
        "IP_Time": "IP",
        "LR_Time": "LP Relaxation",
        "Heuristic_Time": "Heuristic",
        "NaiveHeuristic_Time": "Naive"
    })
    sns.boxplot(
        data=df_time, x="ScenarioID", y="Time (s)", hue="Method",
        palette={key: color_palette[key] for key in df_time["Method"].unique()}
    )
    plt.yscale("log")
    plt.title("Computation Time per Scenario (Log Scale)")
    plt.ylabel("Time (log scale, seconds)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "computation_times.png"), dpi=300)
    plt.show()

    # 3. 📦 Optimality Gaps to IP
    """plt.figure(figsize=(12, 6))
    df_gap = df.copy()
    df_gap["Heuristic vs IP"] = 100 * (df_gap["Heuristic_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]
    df_gap["Naive vs IP"] = 100 * (df_gap["NaiveHeuristic_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]
    df_gap["LP vs IP"] = 100 * (df_gap["LR_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]

    df_gap_melted = df_gap.melt(
        id_vars=["ScenarioID"],
        value_vars=["Heuristic vs IP", "Naive vs IP", "LP vs IP"],
        var_name="Method", value_name="Gap (%)"
    )

    sns.boxplot(
        data=df_gap_melted, x="ScenarioID", y="Gap (%)", hue="Method",
        palette={key: color_palette[key] for key in df_gap_melted["Method"].unique()}
    )
    plt.title("Optimality Gaps to IP per Scenario")
    plt.ylabel("Gap (%)")
    plt.xlabel("Scenario ID")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimality_gaps_to_ip.png"), dpi=300)
    plt.show()"""

    # 📍 Nur Mittelwerte (Punkte) + Automatisch angepasste Broken Axis

    # Gap-Mittelwerte berechnen
    df_gap = df.copy()
    df_gap["Heuristic vs IP"] = 100 * (df_gap["Heuristic_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]
    df_gap["Naive vs IP"] = 100 * (df_gap["NaiveHeuristic_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]
    df_gap["LP vs IP"] = 100 * (df_gap["LR_ObjVal"] - df_gap["IP_ObjVal"]) / df_gap["IP_ObjVal"]

    df_gap_melted = df_gap.melt(
        id_vars=["ScenarioID"],
        value_vars=["Heuristic vs IP", "Naive vs IP", "LP vs IP"],
        var_name="Method", value_name="Gap (%)"
    )
    df_gap_mean = df_gap_melted.groupby(["ScenarioID", "Method"])["Gap (%)"].mean().reset_index()

    # Farben
    color_palette = {
        "Heuristic vs IP": "yellow",
        "Naive vs IP": "red",
        "LP vs IP": "green"
    }

    # Dynamische Achsengrenzen berechnen
    min_gap = df_gap_mean["Gap (%)"].min()
    max_gap = df_gap_mean["Gap (%)"].max()

    # Trennwert automatisch zwischen niedrigem und hohem Bereich setzen
    gap_threshold = 1.0 if max_gap > 5 else 0.1  # anpassbar nach Bedarf

    # Plot-Struktur mit Broken Axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), height_ratios=[1, 4])

    # Obere Achse (nur Werte über Schwelle)
    for method in df_gap_mean["Method"].unique():
        sub = df_gap_mean[(df_gap_mean["Method"] == method) & (df_gap_mean["Gap (%)"] > gap_threshold)]
        ax1.scatter(sub["ScenarioID"], sub["Gap (%)"], label=method,
                    color=color_palette[method], s=80)

    ax1.set_ylim(gap_threshold, max_gap + 0.1 * max_gap)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Gap (%) (high)")

    # Untere Achse (nur Werte unter Schwelle)
    for method in df_gap_mean["Method"].unique():
        sub = df_gap_mean[(df_gap_mean["Method"] == method) & (df_gap_mean["Gap (%)"] <= gap_threshold)]
        ax2.scatter(sub["ScenarioID"], sub["Gap (%)"], label=method,
                    color=color_palette[method], s=80)

    # Dynamische Grenzen für die untere Achse (nur LP & Heuristic)
    min_lp_gap = df_gap_mean[df_gap_mean["Method"] == "LP vs IP"]["Gap (%)"].min()
    max_heur_gap = df_gap_mean[df_gap_mean["Method"] == "Heuristic vs IP"]["Gap (%)"].max()

    padding = 0.1 * (max_heur_gap - min_lp_gap)  # etwas Luft

    ax2.set_ylim(min_lp_gap - padding, max_heur_gap + padding)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel("Gap (%) (low)")
    ax2.set_xlabel("Scenario ID")

    # Legende
    ax1.legend(title="Method")

    # Zacken zwischen Achsen
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([.5, .5], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([.5, .5], [1, 0], transform=ax2.transAxes, **kwargs)

    # Titel & Layout
    plt.suptitle("Mean Optimality Gaps to IP per Scenario (Auto Broken Axis)", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    # Speichern
    plt.savefig(os.path.join(output_dir, "optimality_gap_means_splitaxis_auto.png"), dpi=300)
    plt.show()

# 🔧 Run
plot_results_from_excel("/Users/maximilian/PycharmProjects/OR Midterm Project/generated_structured_20250424_160446/instance_results_summary.xlsx")