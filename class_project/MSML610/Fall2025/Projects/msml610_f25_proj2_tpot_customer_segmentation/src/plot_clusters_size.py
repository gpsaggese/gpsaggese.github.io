import pandas as pd
import matplotlib.pyplot as plt

def plot_cluster_sizes(df, output_path):
    # Count cluster sizes
    counts = df["cluster"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#4c72b0", "#dd8452"])
    plt.title("Customer Cluster Sizes", fontsize=14)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150)
    plt.close()
