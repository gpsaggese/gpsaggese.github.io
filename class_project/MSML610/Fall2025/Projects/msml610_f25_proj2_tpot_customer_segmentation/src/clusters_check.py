import pandas as pd

df = pd.read_csv("outputs/customer_segments.csv")
print(df["cluster"].value_counts())
