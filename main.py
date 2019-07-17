import pandas as pd 

# found dataset on Kaggle
df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

print(df.head())

# we only want taxi ride from NYC to NYC !
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

# main airport of NYC
landmarks = {
    'JFK Airport': (-73.78, 40.643),
    'Laguardia Airport': (-73.87, 40.77),
    'Midtown': (-73.98, 40.76),
    'Lower Manhattan': (-74.00, 40.72),
    'Upper Manhattan': (-73.94, 40.82),
    'Brooklyn': (-73.95, 40.66)
}

df2 = df.copy(deep=True) # we never want to overwrite the original dataframe

def data_cleaning():
    for long in ['pickup_longitude', 'dropoff_longitude']:
        df2 = df2[(df2[long] > nyc_min_longitude) & (df2[long] < nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        df2 = df2[(df2[lat] > nyc_min_latitude) & (df2[lat] < nyc_max_latitude)]   
    return df2

df2 = data_cleaning()