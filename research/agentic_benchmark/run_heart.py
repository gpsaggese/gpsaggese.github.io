import sys
import agentic_bench as ab

datasets = {"Heart_Disease": ab.load_heart("/Users/shobha/Downloads/archive/heart_disease_cleveland.csv")}

if __name__ == "__main__":
    out = "outputs/results_heart.csv"
    ab.run_experiment(
        datasets,
        seeds=range(10),            # tiny test set, so use more seeds
        conditions=[(0.0, 0.0), (0.05, 0.02), (0.15, 0.05), (0.30, 0.05)],
        use_autogluon=("--ag" in sys.argv),
        out_csv=out,
        verbose=True,
    )
    print(ab.summarize(out))
    print(ab.detector_table(out))