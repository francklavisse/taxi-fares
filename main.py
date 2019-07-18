import pandas as pd
import vizualisation as graphic

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

def data_serialize(df2):    
    df2['year'] = df2['pickup_datetime'].dt.year
    df2['month'] = df2['pickup_datetime'].dt.month
    df2['day'] = df2['pickup_datetime'].dt.day
    df2['day_of_week'] = df2['pickup_datetime'].dt.dayofweek
    df2['hour'] = df2['pickup_datetime'].dt.hour
    return df2

df2 = data_cleaning()
df2 = data_serialize(df2)

graphic.plot_lat_long(df2)
graphic.hist_rides_by_day(df2)
graphic.hist_rides_by_hour(df2)

