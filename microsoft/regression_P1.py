import pandas as pd

# load the training dataset
# wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv
bike_data = pd.read_csv('daily-bike-share.csv')
print(bike_data.head())

"""
instant: A unique row identifier
dteday: The date on which the data was observed - in this case, the data was collected daily; so there's one row per date.
season: A numerically encoded value indicating the season (1:spring, 2:summer, 3:fall, 4:winter)
yr: The year of the study in which the observation was made (the study took place over two years - year 0 represents 2011, and year 1 represents 2012)
mnth: The calendar month in which the observation was made (1:January ... 12:December)
holiday: A binary value indicating whether or not the observation was made on a public holiday)
weekday: The day of the week on which the observation was made (0:Sunday ... 6:Saturday)
workingday: A binary value indicating whether or not the day is a working day (not a weekend or holiday)
weathersit: A categorical value indicating the weather situation (1:clear, 2:mist/cloud, 3:light rain/snow, 4:heavy rain/hail/snow/fog)
temp: The temperature in celsius (normalized)
atemp: The apparent ("feels-like") temperature in celsius (normalized)
hum: The humidity level (normalized)
windspeed: The windspeed (normalized)
rentals: The number of bicycle rentals recorded.
"""

# Add number day
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
# print(bike_data.head(32))

# Some general analysis
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
general_analysis = bike_data[numeric_features + ['rentals']].describe()

# Visualize
import matplotlib.pyplot as plt

# This ensures plots are displayed inline in the Jupyter notebook
# %matplotlib inline

# Get the label column
label = bike_data['rentals']


# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Plot the histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')

# Show the figure
# fig.show()

# Plot a histogram for each numeric feature
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
# plt.show()

# plot a bar plot for each categorical feature count
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
# plt.show()

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
# plt.show()

for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    bike_data.boxplot(column = 'rentals', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Bike Rentals")
# plt.show()






