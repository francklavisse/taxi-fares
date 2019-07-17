import pandas as pd
from vizualisation import plot_lat_long

# found dataset on Kaggle
df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

print(df.head())

# we only want taxi ride from NYC to NYC !
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

def data_cleaning():
    dfcopy = df.copy(deep=True)
    for long in ['pickup_longitude', 'dropoff_longitude']:
        dfcopy = dfcopy[(dfcopy[long] > nyc_min_longitude) &
                        (dfcopy[long] < nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        dfcopy = dfcopy[(dfcopy[lat] > nyc_min_latitude) &
                        (dfcopy[lat] < nyc_max_latitude)]
    return dfcopy

df2 = data_cleaning()

plot_lat_long(df2)