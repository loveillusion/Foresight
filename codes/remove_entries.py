import pandas as pd

df = pd.read_csv('in.csv')

is_0 = df['Population'] != 0
df_0 = df[is_0]
df_0.to_csv('out.csv')

df_0.plot()
