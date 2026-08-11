import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import yaml
test_size = yaml.safe_load(open('params.yaml', 'r'))['data_ingestion']['test_size']
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.drop(columns=['tweet_id'], inplace=True)

final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)

test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)