import datetime
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from hepml.core import plot_regression_tree
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#Define SMAPE
def SMAPE(actual, predicted):
    return 1/len(actual) * np.sum(2 * np.abs(predicted-actual) / (np.abs(actual) + np.abs(predicted))*100)

#Load teros data
print("Loading teros Data")
teros_files = glob.glob("rocket4/TEROSoutput*.csv")
X = pd.DataFrame()
for f in teros_files:
  try:
    csv = pd.read_csv(f, index_col=False).dropna()
    X = pd.concat([X, csv])
  except:
    continue
   
#Load power data
print("Loading power Data")
power_files = glob.glob("rocket4/soil*.csv")
y = pd.DataFrame()
for f in power_files:
  try:
    csv = pd.read_csv(f, on_bad_lines='skip', skiprows=10).dropna(how='all')
    csv = csv.rename({'Unnamed: 0': 'timestamp'}, axis='columns')
    y = pd.concat([y,csv])
  except:
    continue
y["timestamp"] = y["timestamp"].round(decimals = 1)

#Sort data by timestamp, convert to datetime
X = X.sort_values(['timestamp'])
y = y.sort_values(['timestamp'])
print("Sorting py timestamp")
X['timestamp'] = pd.to_datetime(X['timestamp'], unit='s')
y['timestamp'] = pd.to_datetime(y['timestamp'], unit='s')

#Merge data by timestamp
print("Merging Data")
uncut_df = pd.merge_asof(left=X,right=y,direction='nearest',tolerance=pd.Timedelta('0.1 min'), on = 'timestamp').dropna(how='all')

#Isolate data from cell0
df = uncut_df.loc[uncut_df['sensorID'] == 0]

#Use only data from after deployment date
df = df.loc[df['timestamp'] > '2021-06-11']

#Calculate power
df["power"] = np.abs(np.multiply(df.iloc[:, 8]*10E-12, df.iloc[:, 9]*10E-9))


#Add power time series
df['previous_power - 1'] = df['power'].shift(1).dropna()
#df['previous_power - 2'] = df['power'].shift(2).dropna()
#df['previous_power - 3'] = df['power'].shift(3).dropna()
#df['previous_power - 4'] = df['power'].shift(4).dropna()

df = df.dropna()

#Re-split data for training
X = pd.concat([df.iloc[:, 0:1], df.iloc[:, 2:5], df.iloc[:, 14:18]], axis = 1).dropna()
y = df.iloc[:, 13:14].dropna()

#Convert datetime to timestamp for training
X["timestamp"] = X["timestamp"].values.astype("float64")

#Creating training and testing sets
X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)
y_train, y_test = train_test_split(y, test_size=0.3, shuffle=False)

#Train model
model = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42)
print("Training Model")
model.fit(X_train, y_train.values.ravel())

#Evaluate SMAPE
print("Train SMAPE:\n", SMAPE(y_train.values.ravel(), model.predict(X_train)))
print("Test SMAPE:\n", SMAPE(y_test.values.ravel(), model.predict(X_test)))
