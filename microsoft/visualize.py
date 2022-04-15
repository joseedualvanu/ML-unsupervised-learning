import pandas as pd
from matplotlib import pyplot as plt

# Load data
df_bitcoin = pd.read_csv('bitcoin_V2.csv',delimiter=',',header='infer')

# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Change datatype
# df_bitcoin = df_bitcoin.astype({"Data": datetime}, errors='raise') 
# change to date
df_bitcoin['Date']= pd.to_datetime(df_bitcoin['Date'])

# Group by month 
# Adj_Close_month1 = df_bitcoin.resample(rule='M', on='Date')['Adj_Close'].mean()
Adj_Close_month1 = df_bitcoin.groupby('Date')['Adj_Close'].mean()
# Adj_Close_month1 = df_bitcoin.groupby([df_bitcoin['Date'][:].month, df_bitcoin['Date'][:].day], as_index=True).mean()
# Adj_Close_month1 = df_bitcoin.groupby(df_bitcoin['Date'][:].month, as_index=True).mean()

# Print info
print(Adj_Close_month1.head())

# Create a bar plot of name vs grade
# plt.bar(x=Adj_Close_month1['Date'], height=Adj_Close_month1['Adj_Close'])
# change to index and : because it is a serie
# ax[0].plt.bar(x=Adj_Close_month1.index, height=Adj_Close_month1[:])
ax[0].bar(x=Adj_Close_month1.index, height=Adj_Close_month1[:])

# Display the plot
# plt.show()

# Create a bar plot of name vs grade
# plt.bar(x=Adj_Close_month1['Date'], height=Adj_Close_month1['Adj_Close'], color='orange')
# change to index and : because it is a serie
# ax[1].plt.bar(x=Adj_Close_month1.index, height=Adj_Close_month1[:], color='orange')
ax[1].bar(x=Adj_Close_month1.index, height=Adj_Close_month1[:], color='orange')

# Customize the chart
# ax[1].plt.title('Bitcoin Price changing over time')
# ax[1].plt.xlabel('Date (dd/mm/yyyy)')
# ax[1].plt.ylabel('Price (USD)')
# ax[1].plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
# ax[1].plt.xticks(rotation=90)
ax[0].set_title('Bitcoin Price changing over time1')
ax[1].set_title('Bitcoin Price changing over time2')

# ax[1].set_xticklabels('Date (dd/mm/yyyy)')
# ax[1].set_yticklabels('Price (USD)')
# ax[1].grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
fig.suptitle('Bitcoin Price Year 2021')
# Display the plot
plt.show()

