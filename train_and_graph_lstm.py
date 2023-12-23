batch = 5 #set batch size for training
weight = 24 #parameterize weighted loss function (see papaer section 3.2.1 for details)
epochs = 25 #set number of training epochs
tstep = '60min' #set size of prediction timestep, ie. 3min for 3 minutes, 60min for 1 hour
sec = 3600 #set the number of seconds in the desired timestep, ie. 180 for 3 minutes, 3600 for 1 hour

%matplotlib inline
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
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.optimizers import SGD
import csv
from collections import defaultdict
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from datetime import datetime

#Load teros data
teros_files = glob.glob("rocket4/TEROSoutput*.csv")
X = pd.DataFrame()
for f in teros_files:
  try:
    csv = pd.read_csv(f, index_col=False).dropna()
    X = pd.concat([X, csv])
  except:
    continue

#Load power data
power_files = glob.glob("rocket4/soil*.csv")
y = pd.DataFrame()
for f in sorted(power_files, key=lambda x: int(x.split('.')[0].split('_')[-1])):
#in power_files:
  try:
    csv = pd.read_csv(f, on_bad_lines='skip', skiprows=10).dropna(how='all')
    csv = csv.rename({'Unnamed: 0': 'timestamp'}, axis='columns')
    y = pd.concat([y,csv])
  except:
    continue
y["timestamp"] = y["timestamp"].round(decimals = 1)

#Convert current to amps, voltage to volts
y["I1L [10pA]"] = np.abs(y["I1L [10pA]"] * 1E-11)
y["V1 [10nV]"] = np.abs(y["V1 [10nV]"] * 1E-8)
y["I1H [nA]"] = np.abs(y["I1H [nA]"] * 1E-9)

#Sort data by timestamp, convert to datetime
X = X.sort_values(['timestamp'])
y = y.sort_values(['timestamp'])
X['timestamp'] = pd.to_datetime(X['timestamp'], unit='s')
y['timestamp'] = pd.to_datetime(y['timestamp'], unit='s')

#Merge data by timestamp
uncut_df = pd.merge_asof(left=X,right=y,direction='nearest',tolerance=pd.Timedelta('1 sec'), on = 'timestamp').dropna(how='all')

#Isolate data from cell0
df = uncut_df.loc[uncut_df['sensorID'] == 0]

#Localize timestamp
df.timestamp = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

#Use only data from after deployment date
#df = df.loc[(df['timestamp'] > '2021-09-24') & (df['timestamp'] < '2021-10-15')] #Future of Clean Computing Graph
#df = df.loc[(df['timestamp'] > '2021-06-24') & (df['timestamp'] < '2021-07-02')]
#df = df.loc[(df['timestamp'] > '2021-06-18')] #Two weeks after deployment
df = df.loc[(df['timestamp'] > '2021-06-04')] #Deployment date
#df = df.loc[(df['timestamp'] > '2021-06-25') & (df['timestamp'] < '2021-06-26')] #Small training set

#Power drop
#df = df.loc[(df['timestamp'] > '2021-11-01') & (df['timestamp'] < '2021-11-22')]

#Drop data outages
df = df.drop(df[(df.timestamp > '2021-11-11') & (df.timestamp < '2021-11-22 01:00:00')].index)
df = df.drop(df[(df.timestamp > '2022-01-27')].index)
df = df[:-1]

#Select the size of prediction timestep, ie. 3 min for 3 minutes, 60min for 1 hour
df = df.resample(tstep, on='timestamp').mean().iloc[1: , :]

#Finish data prep and train model

#Get time since deployement
df['tsd'] = (df.index - df.index[0]).days
df['hour'] = (df.index).hour

#Calculate power
df["power"] = np.abs(np.multiply(df.iloc[:, 7], df.iloc[:, 8]))

#Convert to nW
df['power'] = df['power']*1E9

#Convert to 10 nanoamps, 10 microvolts
df["I1L [10pA]"] = np.abs(df["I1L [10pA]"] * 1E8)
df["V1 [10nV]"] = np.abs(df["V1 [10nV]"] * 1E5)
df["I1H [nA]"] = np.abs(df["I1H [nA]"] * 1E8)

#Add power time series
df['power - 1h'] = df['power'].shift(1).dropna()
df['power - 2h'] = df['power'].shift(2).dropna()
df['power - 3h'] = df['power'].shift(3).dropna()

#Add teros time series
df['EC - 1h'] = df['EC'].shift(1).dropna()
df['EC - 2h'] = df['EC'].shift(2).dropna()
df['EC - 3h'] = df['EC'].shift(3).dropna()

df['temp - 1h'] = df['temp'].shift(1).dropna()
df['temp - 2h'] = df['temp'].shift(2).dropna()
df['temp - 3h'] = df['temp'].shift(3).dropna()

df['raw_VWC - 1h'] = df['raw_VWC'].shift(1).dropna()
df['raw_VWC - 2h'] = df['raw_VWC'].shift(2).dropna()
df['raw_VWC - 3h'] = df['raw_VWC'].shift(3).dropna()

#Add voltage and current time series
df['V1 - 1h'] = df['V1 [10nV]'].shift(1).dropna()
df['V1 - 2h'] = df['V1 [10nV]'].shift(2).dropna()
df['V1 - 3h'] = df['V1 [10nV]'].shift(3).dropna()

df['I1L - 1h'] = df['I1L [10pA]'].shift(1).dropna()
df['I1L - 2h'] = df['I1L [10pA]'].shift(2).dropna()
df['I1L - 3h'] = df['I1L [10pA]'].shift(3).dropna()

df['I1H - 1h'] = df['I1H [nA]'].shift(1).dropna()
df['I1H - 2h'] = df['I1H [nA]'].shift(2).dropna()
df['I1H - 3h'] = df['I1H [nA]'].shift(3).dropna()
df = df.dropna()

#Rename columns
df = df.rename(columns={'I1L [10pA]': 'I1L [μA]', 'V1 [10nV]' : 'V1 [mV]'})

#Prepare train/test datasets
X_train, X_test = train_test_split(pd.concat([df["power - 1h"], df["power - 2h"], df["power - 3h"], df["V1 - 1h"], df["V1 - 2h"], df["V1 - 3h"], df["I1L - 1h"], df["I1L - 2h"], df["I1L - 3h"],df["EC - 1h"], df["EC - 2h"], df["EC - 3h"], df["raw_VWC - 1h"], df["raw_VWC - 2h"], df["raw_VWC - 3h"], df["temp - 1h"], df["temp - 2h"], df["temp - 3h"], df["tsd"], df["hour"]], axis = 1), test_size=0.3, shuffle=False)
y_train, y_test = train_test_split(pd.concat([df["power"], df['V1 [mV]'], df['I1L [μA]']], axis = 1), test_size=0.3, shuffle=False)

#reshape data
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

def custom_loss(y_true, y_pred):
    residual = (y_true - y_pred*1.11).astype("float")
    loss = tf.where(residual < 0, (residual**2) * weight, residual**2)
    return loss

#Build model
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='linear'))
#opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss=custom_loss, metrics=['mape'])

#Train model
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch, verbose=2, shuffle=False)

train_pred = model.predict(X_train, batch_size=batch)
test_pred = model.predict(X_test, batch_size=batch)

#Load previously trained model
model = load_model("drive/MyDrive/jLab Shared Docs/MFC Modeling/lstm7_3min_weighted", custom_objects={ 'custom_loss': custom_loss})
train_pred = model.predict(X_train, batch_size=batch)
test_pred = model.predict(X_test, batch_size=batch)

#Simulate application runtime, evaluate model performance
import math
from matplotlib import pyplot as plt

def internal_R_v3(R=2000): #return internal resistance of v3 cells in ohms
    #https://www.jstage.jst.go.jp/article/jwet/20/1/20_21-087/_pdf
    v0_oc = 48.5e-3 #48.5 mV
    v0_cc = 4.8e-3
    v0_r = R*((v0_oc/v0_cc)-1)

    v1_oc = 43.8e-3
    v1_cc = 20.9e-3
    v1_r = R*((v1_oc/v1_cc)-1)

    v2_oc = 45.2e-3
    v2_cc = 23.5e-3
    v2_r = R*((v2_oc/v2_cc)-1)

    return (v0_r+v1_r+v2_r)/3

def internal_R_v0(R=2000): #return internal resistance of v0 cells in ohms
    v3_oc = 41.7e-3 #41.7mV
    v3_cc = 5.1e-3
    v3_r = R*((v3_oc/v3_cc)-1)

    v4_oc = 48.7e-3
    v4_cc = 16.8e-3
    v4_r = R*((v4_oc/v4_cc)-1)

    v5_oc = 39.1e-3
    v5_cc = 16.9e-3
    v5_r = R*((v5_oc/v5_cc)-1)

    return (v3_r+v4_r+v5_r)/3

def SMFC_current(v, R):
    return v/R

#MODEL
def cap_leakage(E_cap_tn, timestep):
    #Spec for KEMET T491
    return 0.01e-6 * E_cap_tn * timestep

def Matrix_Power(V, R):
    #efficiency interpolated from https://www.analog.com/media/en/technical-documentation/data-sheets/ADP5091-5092.pdf
    #given I_in = 100 uA and SYS = 3V
    #V is the voltage (V) of the SMFC we captured
    #R is the resistance (ohms) of the load we used to get that voltage trace
    #Eta = -292.25665*V**4 + 784.30311*V**3 - 770.71691*V**2 + 342.00502*V + 15.83307
    #Eta = Eta/100
    Eta = 0.60
    Pmax = (V**2)/R
    Pout = Eta*Pmax
    #assert((Eta > 0) & (Eta < 1))
    assert(Pout < 12000e-6)
    return Pout

def update_capEnergy(e0, V_applied, R, C, dt):
    # e0: initial energy stored
    # V_applied: voltage from SMFC
    # R: internal resistance of SMFC
    # C: capacitance of capacitor
    # dt: time step since last data point
    e_cap = e0 + Matrix_Power(V_applied, R)*dt #- cap_leakage(e0, dt)
    v_cap = math.sqrt(2*e_cap/C)
    if e_cap < 0: #Not charging if leakage is greater than energy
        e_cap = 0

    return e_cap, v_cap #output final e and v

def Advanced_energy():
    #Now representing "Advanced"
    #startup time of 2500 ms
    t = 2500e-3
    e = 2.4 * 128e-3 * t
    e_startup = 2.4 * 128e-3 * 5e-3
    return e+e_startup

def Minimal_energy():
    #Now representing "Minimal"
    t = 0.888e-3 #tentative time
    e = 0.9 * 4.8e-3 * t #this uses average current
    e_startup = 0#assume negligible, no known startup time given
    return  e + e_startup

def Analog_energy():
    #Now representing Analog
    t = 1e-3 #estimated operating time
    e = 0.11 * 2.15e-6 * t
    e_startup = 0 #analog device, no startup needed :)
    return e + e_startup

def simulate(t_list, v_list, v_list_pred, C_h):
    # t_list: list of decimal time stamps in unit of days (e.g. 71.85893518518519 day), same length as v_list
    # v_list: list of voltage values from SFMC
    # C_h: capacitance of the capacitor being filled up by harvester
    on_Advanced_list = []
    on_Analog_list = []
    on_Minimal_list = []

    on_Advanced_pred_list = []
    on_Analog_pred_list = []
    on_Minimal_pred_list = []

    #assume capacitor is completely discharged at start
    e_advanced_init = 0
    e_minimal_init = 0
    e_analog_init = 0

    #Initialize sensor reading count
    on_Advanced = 0
    on_Minimal = 0
    on_Analog = 0

    on_Advanced_pred = 0
    on_Minimal_pred = 0
    on_Analog_pred = 0

    cap_energy_analog = []
    cap_energy_minimal = []
    cap_energy_advanced = []

    cap_v_analog = []
    cap_v_minimal = []
    cap_v_advanced = []

    #for each voltage data point
    for jj in range(len(t_list)): #last data point was at 71.85893518518519 day

        #predict amount of energy in capacitor given v0 output
        E_Advanced_pred, v_advanced_pred = update_capEnergy(e_advanced_init, V_applied=v_list_pred[jj], R=2000, C=C_h[0], dt = sec) #set dt as length of prediction interval, in seconds
        E_Minimal_pred, v_minimal_pred = update_capEnergy(e_minimal_init, V_applied=v_list_pred[jj], R=2000, C=C_h[0], dt = sec)
        E_Analog_pred, v_analog_pred = update_capEnergy(e_analog_init, V_applied=v_list_pred[jj], R=2000, C=C_h[0], dt = sec)

        #update actual amount of energy in capacitor given v0 output
        E_Advanced, v_advanced = update_capEnergy(e_advanced_init, V_applied=v_list[jj], R=2000, C=C_h[0], dt = sec)
        E_Minimal, v_minimal = update_capEnergy(e_minimal_init, V_applied=v_list[jj], R=2000, C=C_h[0], dt = sec)
        E_Analog, v_analog = update_capEnergy(e_analog_init, V_applied=v_list[jj], R=2000, C=C_h[0], dt = sec)

        #Predict if we have enough power to turn things on
        if E_Advanced_pred > Advanced_energy():
            on_Advanced_pred = on_Advanced_pred + round(E_Advanced_pred/Advanced_energy())

        if E_Minimal_pred > Minimal_energy():
            on_Minimal_pred = on_Minimal_pred + round(E_Minimal_pred/Minimal_energy())

        if E_Analog_pred > Analog_energy():
            on_Analog_pred = on_Analog_pred + round(E_Analog/Analog_energy())

        #Check if we actually have enough power to turn things on
        if E_Advanced > Advanced_energy():
            on_Advanced = on_Advanced + round(E_Advanced/Advanced_energy())
            E_Advanced = 0 #completely discharge, prob bad assumption will change based on matrix board stat
            v_advanced = 0

        if E_Minimal > Minimal_energy():
            on_Minimal = on_Minimal + round(E_Minimal/Minimal_energy())
            E_Minimal = 0 #completely discharge, prob bad assumption will change based on matrix board stat
            v_minimal = 0

        if E_Analog > Analog_energy():
            on_Analog = on_Analog + round(E_Analog/Analog_energy())
            E_Analog = 0 #completely discharge, prob bad assumption will change based on matrix board stat
            v_analog = 0
        #print(on_Minimal_pred, on_Minimal)

        cap_energy_analog.append(E_Analog)
        cap_energy_minimal.append(E_Minimal)
        cap_energy_advanced.append(E_Advanced)

        cap_v_analog.append(v_analog)
        cap_v_minimal.append(v_minimal)
        cap_v_advanced.append(v_advanced)

        #update start condition for next loop
        e_advanced_init = E_Advanced
        e_minimal_init = E_Minimal
        e_analog_init = E_Analog

        #record the number of sensor reading that day to their respective lists
        on_Advanced_list.append(on_Advanced)
        on_Minimal_list.append(on_Minimal)
        on_Analog_list.append(on_Analog)

        on_Advanced_pred_list.append(on_Advanced_pred)
        on_Minimal_pred_list.append(on_Minimal_pred)
        on_Analog_pred_list.append(on_Analog_pred)

        #Reset sensor reading count
        on_Advanced = 0
        on_Minimal = 0
        on_Analog = 0

        on_Advanced_pred = 0
        on_Minimal_pred = 0
        on_Analog_pred = 0

def getMax(c_list, input_list):
    max_value = max(input_list)
    i = [index for index, item in enumerate(input_list) if item == max_value][0]
    return i, max_value, c_list[i]

def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def getMFC_data(y_test, test_pred):
    unix_time = y_test.index
    d0 = unix_time[0]
    days = []
    for d in unix_time:
        day = d
        day_from_start = day-d0
        decimal_day = day_from_start.total_seconds()/(24 * 3600)
        days.append(decimal_day)

    return days

days  = getMFC_data(y_test, test_pred)
v_avg_true = y_test['V1 [mV]']/1E5
v_avg_pred = test_pred[:, 1]/1E5
C0 = [0.007000000000000006, 0.007000000000000006, 0.007000000000000006]
Advanced_true, Advanced_pred, Minimal_true, Minimal_pred, ignore, ignore2 = simulate(days, v_avg_true, v_avg_pred, C0)

metrics = y_test
metrics = metrics.drop(columns = ['V1 [mV]', 'I1L [μA]'])
metrics['min_active'] = Minimal_true #Track if there is enough energy to activate the device at the end of time interval
metrics['min_active_pred'] = Minimal_pred #Track if there is enough energy to activate the device at the end of time interval, using predicted values
metrics['adv_active'] = Advanced_true #Track if there is enough energy to activate the device at the end of time interval
metrics['adv_active_pred'] = Advanced_pred #Track if there is enough energy to activate the device at the end of time interval, using predicted values

min_active = metrics['min_active']
min_active_pred = metrics['min_active_pred']
false_active_pred = ((metrics['min_active'] < metrics['min_active_pred']) * (metrics['min_active_pred'] - metrics['min_active']))#((metrics['min_active_pred'] > metrics['min_active']) * (metrics['min_active_pred'] - metrics['min_active']))
missed_active_pred = ((metrics['min_active'] > metrics['min_active_pred']) * (metrics['min_active'] - metrics['min_active_pred']))

print('Minimal Application')
print('Total activations:', min_active.sum())
print('Predicted activations:', min_active_pred.sum())
print('False predicted activations: %d, %.3f%%' % (false_active_pred.sum(), false_active_pred.sum() * 100/min_active_pred.sum()))
print('Missed predicted activations: %d, %.3f%%' % (missed_active_pred.sum(), missed_active_pred.sum() * 100/min_active.sum()))
print('Voltage overestimation rate: %.3f%%' % ((y_test['V1 [mV]'].values <= test_pred[:, 1]).mean() * 100))
print("Train MAPE power: %.3f%%" % (MAPE(y_train['power'].values.ravel(), train_pred[:, 0]) * 100))
print("Test MAPE power: %.3f%%" % (MAPE(y_test['power'].values.ravel(), test_pred[:, 0]) * 100))
print("Train MAPE voltage: %.3f%%" % (MAPE(y_train['V1 [mV]'], train_pred[:, 1]) * 100))
print("Test MAPE voltage: %.3f%%" % (MAPE(y_test['V1 [mV]'], test_pred[:, 1]) * 100))
print("Train MAPE current: %.3f%%" % (MAPE(y_train['I1L [μA]'], train_pred[:, 2]) * 100))
print("Test MAPE current: %.3f%%" % (MAPE(y_test['I1L [μA]'], test_pred[:, 2]) * 100))

#Save model
#model.save("lstm7_30min_weighted_mod_over", overwrite=True, save_format=None)

#!mv lstm7_30min_weighted_mod_over 'drive/MyDrive/jLab Shared Docs/MFC Modeling' #Choose directory to save model
