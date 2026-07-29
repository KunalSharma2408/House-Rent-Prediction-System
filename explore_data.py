import pandas as pd
df = pd.read_csv('data.csv')
print("Rent quantiles:")
print(df['rent'].quantile([0.5, 0.9, 0.95, 0.99, 1.0]))
