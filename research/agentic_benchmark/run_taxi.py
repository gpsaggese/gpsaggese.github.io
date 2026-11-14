import sys
import agentic_bench as ab

keep = "--keep-payment" in sys.argv
datasets = {"NYC_Taxi": ab.load_nyc("/Users/shobha/Downloads/yellow_tripdata_2024-01.parquet",
                                    n=5000, keep_payment_type=keep)}

if __name__ == "__main__":
    out = "outputs/results_taxi_keep.csv" if keep else "outputs/results_taxi_drop.csv"
    ab.run_experiment(
        datasets,
        seeds=range(5),
        conditions=[(0.0, 0.0), (0.15, 0.05)],
        out_csv=out,
        verbose=True,
    )
    print(ab.summarize(out))
    print(ab.detector_table(out))