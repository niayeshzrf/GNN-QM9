import pandas as pd
df = pd.read_csv("../data/QM9/raw/gdb9.sdf.csv")

homo_vals = df.iloc[:, 7].values  # index 7 = HOMO
HOMO_MEAN = homo_vals.mean()
HOMO_STD = homo_vals.std()

print(f"HOMO_MEAN: {HOMO_MEAN:.6f}")
print(f"HOMO_STD: {HOMO_STD:.6f}")