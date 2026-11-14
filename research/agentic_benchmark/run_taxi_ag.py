import agentic_bench as ab
from run_taxi import datasets           # payment_type dropped, the fair baseline

if __name__ == "__main__":
    out = "outputs/results_taxi_ag.csv"
    ab.run_experiment(
        datasets,
        seeds=range(5),
        conditions=[(0.0, 0.0), (0.15, 0.05)],
        use_autogluon=True,
        out_csv=out,
        verbose=True,
    )
    print(ab.summarize(out))
    print(ab.detector_table(out))