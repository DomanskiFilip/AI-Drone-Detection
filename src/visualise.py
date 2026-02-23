import pandas
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy
from run_tracker import load_runs

def plot_metrics_over_runs(df: pandas.DataFrame):
    # Line plot of key metrics across runs, grouped by model
    metrics = ["accuracy", "precision", "recall", "f1", "rmse", "r2"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Model Metrics Across Runs", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes.flatten(), metrics):
        seaborn.lineplot(
            data=df,
            x=df.index,
            y=metric,
            hue="model",
            marker="o",
            ax=ax
        )
        ax.set_title(metric.upper())
        ax.set_xlabel("Run #")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("metrics_over_runs.png", dpi=150)
    plt.show()


def plot_model_comparison(df: pandas.DataFrame):
    # Bar chart comparing latest run metrics per model
    # Take the most recent run per model
    latest = df.sort_values("timestamp").groupby("model").last().reset_index()

    # Convert precision, recall, and f1 to percentage for better comparison on the graph
    latest["precision"] = latest["precision"] * 100
    latest["recall"] = latest["recall"] * 100
    latest["f1"] = latest["f1"] * 100

    melted = latest.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    seaborn.barplot(data=melted, x="Metric", y="Score", hue="model", palette="Set2")
    plt.title("Latest Run: Model Comparison")
    plt.ylim(0, 105) # Set y-limit to accommodate percentage values
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()



def plot_confusion_matrices(df: pandas.DataFrame, class_names: list):
    # Plot confusion matrices for the most recent run of each model
    latest = df.sort_values("timestamp").groupby("model").last().reset_index()
    n_models = len(latest)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, latest.iterrows()):
        cm = numpy.array(json.loads(row["confusion_matrix"]))
        seaborn.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_title(f"{row['model']}\nConfusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()


def plot_error_metrics(df: pandas.DataFrame):
    # Box plot of error metrics (MAE, RMSE) distribution per model across all runs
    melted = df.melt(
        id_vars="model",
        value_vars=["mae", "rmse"],
        var_name="Error Metric",
        value_name="Value"
    )
    plt.figure(figsize=(10, 6))
    seaborn.boxplot(data=melted, x="model", y="Value", hue="Error Metric", palette="Set1")
    plt.title("Error Metric Distribution Across Runs")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("error_metrics.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    dataFrame = load_runs()
    if dataFrame.empty:
        print("Run some experiments first!")
    else:
        class_names = ["Birds", "Drones", "Aeroplanes"]
        plot_metrics_over_runs(dataFrame)
        plot_model_comparison(dataFrame)
        plot_confusion_matrices(dataFrame, class_names)
        plot_error_metrics(dataFrame)