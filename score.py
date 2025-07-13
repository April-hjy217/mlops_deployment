#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pickle                    # For loading the saved model
import pandas as pd             # Data manipulation
import numpy as np              # Numerical operations
import click                    # CLI parsing

@click.command()
@click.option(
    '--year', '-y',
    type=int,
    default=2023,
    help='Year of the data to score, e.g. 2023'
)
@click.option(
    '--month', '-m',
    type=int,
    default=3,
    help='Month of the data to score (1–12), e.g. 3 for March'
)
def main(year, month):
    """
    Main entry point:
      - Build input URL and output filename from year/month
      - Load the DictVectorizer and regression model
      - Download & preprocess the Parquet data
      - Predict durations
      - Create ride_id values
      - Save the results to a Parquet file
      - Print the standard deviation of predictions and the file size
    """
    # 1. Construct input URL and output file name
    input_url = (
        f'https://d37ci6vzurychx.cloudfront.net/trip-data/'
        f'yellow_tripdata_{year:04d}-{month:02d}.parquet'
    )
    output_file = f'results_{year:04d}_{month:02d}.parquet'

    # 2. Load the vectorizer and model from a pickle
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # 3. Download and preprocess the raw data
    df = pd.read_parquet(input_url, engine='pyarrow')
    # Compute ride duration in minutes
    df['duration'] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    # Filter out durations outside [1, 60] minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert location IDs to strings for DictVectorizer
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    # 4. Transform features and predict
    feature_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X = dv.transform(feature_dicts)
    y_pred = model.predict(X)

    mean_pred = np.mean(y_pred)
    click.echo(f"Prediction mean duration: {mean_pred:.2f}")

    # 5. Generate ride_id in format YYYY/MM_index
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)

    # 6. Build results DataFrame and save to Parquet
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # 7. Print statistics
    std_dev = np.std(y_pred)
    size_mb = os.path.getsize(output_file) / (1024 ** 2)
    click.echo(f"Prediction standard deviation: {std_dev:.2f}")
    click.echo(f"Output file size: {size_mb:.1f} MB")

if __name__ == '__main__':
    main()
