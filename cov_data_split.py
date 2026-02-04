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
    last_index = {i:0 for i,_ in counts.items()}
    print(f'Amount test = {amount_test}, Amount train = {amount_train}\n')

    # filling dataframes
    for current_data in [[amount_train, train_df], [amount_test, test_df]]:
        # iterating over Cover_Type, adding the needed amount of each one in, then recording the index at which it stopped to resume for the next one
        for i in last_index:
            single_covtype_df = df[df['Cover_Type']==i]  

            start_row = last_index[i]
            end_row = start_row+current_data[0]
            rows_to_copy = single_covtype_df.iloc[start_row:end_row]
            print(rows_to_copy.shape[0])
            last_index[i] = end_row+1

            current_data[1] = pd.concat([current_data[1], rows_to_copy], ignore_index=True)

    # Checking output
    print(train_df.head())
            



main()
