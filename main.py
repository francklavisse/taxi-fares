import pandas as pd
import os 
import vizualisation as graphic
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

os.environ['KMP_DUPLICATE_LIB_OK']='True'

airports = {
    'JFK_Airport': (-73.78, 40.643),
    'Laguardia_Airport': (-73.87, 40.77),
    'Newark_Airport': (-74.18, 40.69)
}

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

    print(dfcopy.isnull().sum()) # result = 0.001% of missing data
    dfcopy = dfcopy.dropna()
    dfcopy = dfcopy[(dfcopy['fare_amount'] >= 0) & (dfcopy['fare_amount'] <= 100)]
    dfcopy.loc[dfcopy['passenger_count'] == 0, 'passenger_count'] = 1
    return dfcopy

def euclidian_distance(lat1, long1, lat2, long2): # get distance between 2 points
    return (((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5)

def airport_distance(df2):    
    for airport in airports:
        df2['pickup_dist_' + airport] = euclidian_distance(df2['pickup_latitude'], df2['pickup_longitude'], airports[airport][1], airports[airport][0])
        df2['dropoff_dist_' + airport] = euclidian_distance(df2['dropoff_latitude'], df2['dropoff_longitude'], airports[airport][1], airports[airport][0])
    return df2

def feature_engineering(df2):    
    # convert date to numerical data
    df2['year'] = df2['pickup_datetime'].dt.year
    df2['month'] = df2['pickup_datetime'].dt.month
    df2['day'] = df2['pickup_datetime'].dt.day
    df2['day_of_week'] = df2['pickup_datetime'].dt.dayofweek
    df2['hour'] = df2['pickup_datetime'].dt.hour
    
    df2.drop(['pickup_datetime', 'key'], axis=1, inplace=True) # drop useless column

    df2['distance'] = euclidian_distance(df2['pickup_latitude'], df2['pickup_longitude'], df2['dropoff_latitude'], df2['dropoff_longitude'])
    df2 = airport_distance(df2)
    return df2    

def scale_df(df2):
    df_prescaled = df2.copy()
    df_scaled = df2.drop(['fare_amount'], axis=1)
    df_scaled = scale(df_scaled)
    cols = df2.columns.tolist()
    cols.remove('fare_amount')
    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df2.index)
    df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
    return df_prescaled, df_scaled.copy()


df2 = data_cleaning()
df2 = feature_engineering(df2)

# graphic.plot_lat_long(df2)
# graphic.hist_rides_by_day(df2)
# graphic.hist_rides_by_hour(df2)
# graphic.hist_fares(df2)
# graphic.hist_passenger_count(df2)
# graphic.scatter_fare_distance(df2)

# print(df2[['key', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_dist_JFK_Airport', 'dropoff_dist_JFK_Airport']].head())

df_prescaled, df_scaled = scale_df(df2)

X = df_scaled.loc[:, df.columns != 'fare_amount']
y = df_scaled.loc[:, 'fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=['mse'])
model.fit(X_train, y_train, epochs=1)