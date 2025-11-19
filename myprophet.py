from datetime import datetime
import sys
import pandas as pd
import numpy as np
import time

xcols = ["store_id","product_id","datestamp","items_sold"]

dfIn = pd.read_csv(sys.stdin, sep="\t", header=None, names= xcols,
                   index_col=False, iterator=False, 
                   dtype = {  "items_sold" : np.int64 },
                   parse_dates=["datestamp"]
                  )



# For AMPs that receive no data, exit the script instance gracefully.
if dfIn.empty:
    sys.exit()

from prophet import Prophet

def predict_sales(sales_df, future_days=28):
    predictions = []
    store_product_combinations = sales_df[['store_id', 'product_id']].drop_duplicates()
    
    for _, row in store_product_combinations.iterrows():
        store_id = row['store_id']
        product_id = row['product_id']
        
        # Filter data for the current store and product
        df = sales_df[(sales_df['store_id'] == store_id) & (sales_df['product_id'] == product_id)]
        df = df[['datestamp', 'items_sold']].rename(columns={'datestamp': 'ds', 'items_sold': 'y'})

        # Fit the Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df)

        # Create future dataframe for predictions
        future = model.make_future_dataframe(periods=future_days)
        
        # Predict future sales
        forecast = model.predict(future)
        
        # Select only the future days and relevant columns
        forecast = forecast[['ds', 'yhat']].tail(future_days)
        forecast['store_id'] = store_id
        forecast['product_id'] = product_id
        
        predictions.append(forecast)

    # Combine all predictions into a single DataFrame
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.rename(columns={'ds': 'datestamp', 'yhat': 'predicted_items_sold'}, inplace=True)
    
    return predictions_df


sales_pred_df = predict_sales(dfIn.sort_values(['store_id', 'product_id','datestamp']))

for _, row in sales_pred_df.iterrows():
    print('\t'.join(map(str, row.values)))
