import pandas as pd
import numpy as np
import os

def main():
    print("Main started...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trainingDataPath = os.path.join(base_dir, "Data", "covtype", "covtype.csv")

    # Setting some values
    amount_test = 0.05
    amount_train = 0.3

    # Configuring dataframe
    df = pd.read_csv(trainingDataPath)
    print(df.head())
    total_length = df.shape[0]
    print(f'Number of rows = {total_length}')
    counts = df['Cover_Type'].value_counts()
    print(f"Distinct Counts = {counts}")

    # Creating test dataframe and needed values
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    amount_test = int(amount_test*total_length)+1
    amount_train = int(amount_train*total_length)+1
    last_index = {i:0 for i in df.columns}

    # filling dataframes
    for current_df in [amount_train, amount_test]:
        # iterating over each column, adding the needed amount of each one in, then recording the index at which it stopped to resume for the next one
        for i in last_index:
            print(i)



main()
