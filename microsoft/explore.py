import pandas as pd

# Load data
df_bitcoin = pd.read_csv('bitcoin_V1.csv',delimiter=',',header='infer')

# Print info
print(df_bitcoin.head())

# Nulls?
print(df_bitcoin.isnull().sum())
print(df_bitcoin[df_bitcoin.isnull().any(axis=1)])

# If there null replace with average in Adj_Close
df_bitcoin.Adj_Close = df_bitcoin.Adj_Close.fillna(df_bitcoin.Adj_Close.mean())

# Explore data
mean_adj_close = df_bitcoin['Adj_Close'].mean()
mean_open = df_bitcoin.Open.mean()
min_adj_close = df_bitcoin.Adj_Close.min()
max_adj_close = df_bitcoin.Adj_Close.max()

# Print the mean study hours and mean grade
print('Average bitcoin price: {:.2f}\nAverage bitcoin price open: {:.2f}'.format(mean_adj_close, mean_open))

# Get prices which are higher than mean 
passes = pd.Series(df_bitcoin['Adj_Close'] >= mean_adj_close)
df_bitcoin_higher = pd.concat([df_bitcoin, passes.rename("Adj_Close_Higher_Mean")], axis=1)

# Count higher
print(df_bitcoin.groupby(df_bitcoin_higher.Adj_Close_Higher_Mean).Adj_Close.count())

# Create a DataFrame with the data sorted by Adj_close (descending)
df_bitcoin_sort = df_bitcoin.sort_values('Adj_Close', ascending=True)

