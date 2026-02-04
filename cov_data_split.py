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
    last_index = {i:[0, v/total_length] for i,v in counts.items()}
    print(f'Amount test = {amount_test}, Amount train = {amount_train}\n')

    # filling dataframes
    for current_amount, current_df in zip([amount_train, amount_test], ['train_df', 'test_df']):
        # iterating over Cover_Type, adding the needed amount of each one in, then recording the index at which it stopped to resume for the next one
        temp_df = pd.DataFrame()  

        for i in last_index:
            single_covtype_df = df[df['Cover_Type']==i]  

            start_row = last_index[i][0]
            #print(f"Start index = {start_row}, proportion of this to take = {current_amount}, length of this one = {single_covtype_df.shape[0]}")
            end_row = int(start_row + current_amount * single_covtype_df.shape[0])
            rows_to_copy = single_covtype_df.iloc[start_row:end_row]
            #print(f"Total rows to copy for {i}: {rows_to_copy.shape[0]}")
            last_index[i][0] = end_row+1

            temp_df = pd.concat([temp_df, rows_to_copy], ignore_index=True)

            if current_df == 'train_df':
                train_df = temp_df
            else:
                test_df = temp_df

    # Shuffling all the cov data
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    # Checking output
    print(train_df.shape[0], test_df.shape[0])
            
    # Write to csv
    train_df.to_csv(os.path.join(base_dir, "Data", "covtype", "covtype_train.csv"))
    test_df.to_csv(os.path.join(base_dir, "Data", "covtype", "covtype_test.csv"))

    print("Complete!")

main()