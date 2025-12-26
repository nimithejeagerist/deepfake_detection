from pathlib import Path
import pandas as pd

# Fakes
folder_one = Path("C:/Users/nimza/Documents/dd/fakes")
matches = [str(p.name) for p in folder_one.iterdir()]
labels = [1 for i in range(len(matches))]
df1 = pd.DataFrame(data={'filename': matches, 'labels': labels})

# Real
folder_two = Path("C:/Users/nimza/Documents/dd/real")
matches = [str(p.name) for p in folder_two.iterdir()]
labels = [0 for i in range(len(matches))]
df2 = pd.DataFrame(data={'filename': matches, 'labels': labels})

# Combine dfs and shuffle
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Train test split
df_train = df_combined.sample(frac=0.8, random_state=42)
df_test = df_combined.loc[~df_combined.index.isin(df_train.index)]

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train.to_csv("train.csv")
df_test.to_csv("test.csv")