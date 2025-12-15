import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def create_segments(data, window_size=24, overlap=12):
    # splits a continuous time series into overlapping segments

    segments = []
    step = window_size - overlap
    # ensure we don't go out of bounds
    for i in range(0, len(data) - window_size + 1, step):
        segments.append(data[i: i + window_size])
    return segments


def compute_likelihoods(model, segments):
    # computes the log-likelihood of each segment under the HMM

    scores = []
    for seq in segments:
        try:
            # score() returns the total log-likelihood for the sequence
            scores.append(model.score(seq))
        except:
            # handle edge cases (e.g., NaN) by assigning a very low score
            scores.append(-np.inf)
    return np.array(scores)


def simulate_anomalies(df_test, feature_multiplier=15, num_anomalies=5, window_size=24):
    # injects synthetic anomalies into the test set to evaluate detection

    df_simulated = df_test.copy()
    total_len = len(df_simulated)
    anomaly_starts = []

    # randomly choose start times, ensuring they don't overlap
    possible_starts = np.arange(0, total_len - window_size, window_size * 2)
    if len(possible_starts) < num_anomalies:
        chosen_starts = possible_starts
    else:
        chosen_starts = np.random.choice(possible_starts, num_anomalies, replace=False)

    chosen_starts.sort()

    for start in chosen_starts:
        end = start + window_size
        # inject anomaly by multiply values by some factor (e.g., a massive spike)
        df_simulated.iloc[start:end] = df_simulated.iloc[start:end] * feature_multiplier
        anomaly_starts.append(start)

    return df_simulated, anomaly_starts


def plot_likelihood_distribution(scores, threshold, title="Log-Likelihood Distribution"):
    # plots the histogram of log-likelihood scores and the anomaly threshold

    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, bins=50, label='Segment Scores', color='blue')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title(title)
    plt.xlabel('Log-Likelihood (Lower = More Anomalous)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_anomaly_detection(segment_indices, scores, threshold, true_anomaly_indices=None):
    # visualizes the timeline of scores and highlights detected vs true anomalies

    # create a DataFrame for easy handling in the plot logic
    results_df = pd.DataFrame({
        'log_likelihood_score': scores
    }, index=segment_indices)

    # identify anomalies based on threshold
    anomalies = results_df[results_df['log_likelihood_score'] < threshold]

    plt.figure(figsize=(14, 7))

    # plot all log-likelihood scores as a line (Baseline)
    plt.plot(results_df.index, results_df['log_likelihood_score'],
             label='Segment Log-Likelihood Score',
             color='tab:blue',
             linewidth=1.5,
             alpha=0.8)

    # add the anomaly threshold line
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Anomaly Threshold ({threshold:.2f})')

    # highlight detected anomalies
    plt.scatter(anomalies.index.values,
                anomalies['log_likelihood_score'],
                color='red',
                s=80,
                marker='X',
                zorder=5,
                label=f'Detected Anomalies ({len(anomalies)})')

    # highlight True injected anomalies (aka green zones)
    if true_anomaly_indices is not None:
        # convert indices to a set for easier label handling
        added_label = False
        for start_idx in true_anomaly_indices:
            label = 'True Anomaly Window' if not added_label else None
            # draw a span covering the window (~2 segments width as visual aid)
            plt.axvspan(start_idx, start_idx + 2, color='green', alpha=0.2, label=label)
            added_label = True

    plt.title('Anomaly Detection: Log-Likelihood Scores Over Test Time Segments', fontsize=16)
    plt.xlabel('Time Segment Index', fontsize=14)
    plt.ylabel('Log-Likelihood Score (Normalcy)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()