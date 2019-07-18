import matplotlib.pyplot as plt
import numpy as np

# main airports of NYC
landmarks = {
    'JFK Airport': (-73.78, 40.643),
    'Laguardia Airport': (-73.87, 40.77),
    'Midtown': (-73.98, 40.76),
    'Lower Manhattan': (-74.00, 40.72),
    'Upper Manhattan': (-73.94, 40.82),
    'Brooklyn': (-73.95, 40.66)
}


def draw_landmarks(ax):
    for landmark in landmarks:
        ax.plot(landmarks[landmark][0], landmarks[landmark][1],
                '*', markersize=15, alpha=1, color='r'
                )
        ax.annotate(landmark,
                    (landmarks[landmark][0] + 0.005,
                     landmarks[landmark][1] + 0.005),
                    color='r')


def plot_lat_long(df):
    plt.subplots(2, 1, figsize=(12, 12))

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    draw_landmarks(ax1)
    draw_landmarks(ax2)

    ax1.plot(
        list(df.pickup_longitude),
        list(df.pickup_latitude),
        '.', markersize=1)
    ax1.set_title('Pickup Locations in NYC')
    ax1.grid(None)
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Longitude")

    ax2.plot(
        list(df.dropoff_longitude),
        list(df.dropoff_latitude),
        '.', markersize=1)
    ax2.set_title('Dropoff Locations in NYC')
    ax2.grid(None)
    ax2.set_xlabel("Latitude")
    ax2.set_ylabel("Longitude")

    plt.show()


def hist_rides_by_day(df):
    df['day_of_week'].plot.hist(
        bins=np.arange(8) - 0.5,
        ec='black',
        ylim=(60000, 75000)
    )
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.title('Day of Week Histogram')
    plt.show()

def hist_rides_by_hour(df):
    df['hour'].plot.hist(bins=24, ec='black')
    plt.title('Pickup Hour Histogram')
    plt.xlabel('Hour')
    plt.show()