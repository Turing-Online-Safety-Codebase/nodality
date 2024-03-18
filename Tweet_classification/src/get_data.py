import pandas as pd
import os
import glob


def get_tweets(str):
    complete_dataset = []

    if str == 'mps':
        path = "/MPTweets"
    else:
        path = "/JournalistTweets"
    

    csv_files = glob.glob(os.path.join(path, "*.csv"))

    # loop over the list of csv files
    for f in csv_files:
        # read the csv file
        df = pd.read_csv(f, index_col=None, header=0)

        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d')

        filtered_df = df.loc[(df['created_at'] >= '2022-02-01') & (df['created_at'] <= '2022-01-31')]

        complete_dataset.append(filtered_df)

    dataset = pd.DataFrame(pd.concat(complete_dataset))

    dataset['created_at'] = pd.to_datetime(dataset['created_at'])

    dataset['created_at'] = dataset['created_at'].dt.strftime('%d/%m/%y')

    dataset.rename({'created_at': 'date'}, axis=1, inplace=True)

    # print(dataset['date'])

    return dataset