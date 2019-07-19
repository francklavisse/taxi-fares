import pandas as pd
import os
import vizualisation as graphic
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

airports = {
    'JFK_Airport': (-73.78, 40.643),
    'Laguardia_Airport': (-73.87, 40.77),
    'Newark_Airport': (-74.18, 40.69)
}

# found dataset on Kaggle
df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

print(df.head())


def data_cleaning():
    dfcopy = df.copy(deep=True)
    dfcopy = dfcopy.dropna()
    dfcopy = dfcopy[(dfcopy['fare_amount'] >= 0) &
                    (dfcopy['fare_amount'] <= 100)]
    dfcopy.loc[dfcopy['passenger_count'] == 0, 'passenger_count'] = 1

    # we only want taxi ride from NYC to NYC !
    nyc_min_longitude = -74.05
    nyc_max_longitude = -73.75
    nyc_min_latitude = 40.63
    nyc_max_latitude = 40.85

    for long in ['pickup_longitude', 'dropoff_longitude']:
        dfcopy = dfcopy[(dfcopy[long] > nyc_min_longitude) &
                        (dfcopy[long] < nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        dfcopy = dfcopy[(dfcopy[lat] > nyc_min_latitude) &
                        (dfcopy[lat] < nyc_max_latitude)]

    print(dfcopy.isnull().sum())  # result = 0.001% of missing data
    return dfcopy


def euclidian_distance(lat1, long1, lat2, long2):  # get distance between 2 points
    return (((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5)


def airport_distance(df):
    for airport in airports:
        df['pickup_dist_' + airport] = euclidian_distance(
            df['pickup_latitude'], df['pickup_longitude'], airports[airport][1], airports[airport][0])
        df['dropoff_dist_' + airport] = euclidian_distance(
            df['dropoff_latitude'], df['dropoff_longitude'], airports[airport][1], airports[airport][0])
    return df


def feature_engineering(df):
    # convert date to numerical data
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour

    df = df.drop(['pickup_datetime'], axis=1)

    df['distance'] = euclidian_distance(
        df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df = airport_distance(df)
    df = df.drop(['key'], axis=1)
    return df


def scale_df(df):
    df_prescaled = df.copy()
    df_scaled = df.drop(['fare_amount'], axis=1)
    df_scaled = scale(df_scaled)
    cols = df.columns.tolist()
    cols.remove('fare_amount')
    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
    df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
    return df_prescaled, df_scaled.copy()


df = data_cleaning()
df = feature_engineering(df)

# graphic.plot_lat_long(df2)
# graphic.hist_rides_by_day(df2)
# graphic.hist_rides_by_hour(df2)
# graphic.hist_fares(df2)
# graphic.hist_passenger_count(df2)
# graphic.scatter_fare_distance(df2)

# print(df2[['key', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_dist_JFK_Airport', 'dropoff_dist_JFK_Airport']].head())

df_prescaled, df_scaled = scale_df(df)

X = df_scaled.loc[:, df_scaled.columns != 'fare_amount']
y = df_scaled.fare_amount
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


def predict_random(df_prescaled, X_test, model):
    sample = X_test.sample(
        n=1, random_state=np.random.randint(low=0, high=10000))
    idx = sample.index[0]
    actual_fare = df_prescaled.loc[idx, 'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx, 'day_of_week']]
    hour = df_prescaled.loc[idx, 'hour']
    predicted_fare = model.predict(sample)[0][0]
    rmse = np.sqrt(np.square(predicted_fare - actual_fare))

    print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare))
    print("RMSE: ${:0.2f}".format(rmse))


predict_random(df_prescaled, X_test, model)
