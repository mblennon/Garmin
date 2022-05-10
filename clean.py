from datetime import datetime
import time
import os
import json
import datetime
from collections import ChainMap
from math import exp
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

# Get Json from folder and concat into dataframe

# s2 = pd.json_normalize(d)

base_dir = 'DI_CONNECT/DI-Connect-Fitness/'
data_list = []
for file in os.listdir(base_dir):
    # If file is a json, construct it's full path and open it, append all json data to list
    if 'json' in file:
        json_path = os.path.join(base_dir, file)
        with open(json_path) as f:
            s = json.load(f)
            s1 = s[0]
            s2 = s1["summarizedActivitiesExport"]
            s3 = pd.json_normalize(s2)
            data_list.append(s3)

data = pd.concat(data_list)
data = data.reset_index()
data = data.drop(columns=['index'])

# create new columns with better format

data["duration_min"] = data.duration / 1000 / 60
data["pace_min_per_km"] = (100 / 60) / data["avgSpeed"]
data['distance_metres'] = data['distance'] / 100

# change being timestamp from ms to sec and float

data["begin"] = data["beginTimestamp"].apply(pd.to_numeric) / 1000

# create list of start times in datetime format and append to dataframe

dates = []
for index, row in data.iterrows():
    dates.append(data["begin"][index])

length = len(dates)
i = 0
while i < length:
    # dates[i] = datetime.fromtimestamp(dates[i])
    dates[i] = pd.Timestamp(dates[i], unit='s')
    i += 1

data["startTime"] = dates

# reduce columns

cols = ['activityType', 'startTime', 'distance_metres', 'duration_min', 'avgHr', 'avgSpeed']

data2 = data[cols]

# look at runs only with hr

runs_with_HR = data2[data2['activityType'] == 'running'].dropna(subset=['avgHr'])
runs_with_HR = runs_with_HR.reset_index()


def Trimp_from_Hr(duration, HR, rest, max, sex):
    if sex == 'M':
        var = 1.92
    else:
        var = 1.67
    HRR = (HR - rest) / (max - rest)
    Ti = duration * HRR * 0.64 * exp(var * HRR)
    return Ti


# x = Trimp_from_Hr(runs_with_HR['duration_min'][1], runs_with_HR['avgHr'][1],40,165, 'M')

trimp_values = []
for index, row in runs_with_HR.iterrows():
    trimp_values.append(Trimp_from_Hr(runs_with_HR['duration_min'][index], runs_with_HR['avgHr'][index], 40, 165, 'M'))

runs_with_HR['Trimp'] = trimp_values

runs_with_HR['date'] = runs_with_HR['startTime'].dt.date

runs_with_HR = runs_with_HR.drop(['index'], axis=1)
runs_with_HR = runs_with_HR.set_index('date')
idx = pd.date_range('2012-07-23','2022-04-19')

runs_with_HR2 = runs_with_HR.groupby(runs_with_HR.index).agg({'Trimp': sum})
runs_with_HR3 = runs_with_HR2.reindex(idx, fill_value=0)

# def Banister(trimp, k1, k2, r1, r2):
#     fitness = trimp * exp(-1 / r1)
#     fatigue = trimp * exp(-1 / r2)
#     performance = fitness * k1 - fatigue * k2
#     return fitness, fatigue, performance


trimps = runs_with_HR3.Trimp.tolist()
fitness_values = []
fatigue_values = []
performance_values = []
fitness = 0
fatigue = 0
r1 = 49
r2 = 11
k1 = 1.0
k2 = 1.8
for i in range(len(trimps)):
    fitness = fitness * exp(-1 / r1) + trimps[i]
    fatigue = fatigue * exp(-1 / r2) + trimps[i]
    performance = fitness*k1 - fatigue*k2
    fitness_values.append(fitness)
    fatigue_values.append(fatigue)
    performance_values.append(performance)

result = pd.DataFrame({'date': idx, 'TRIMP': trimps, 'Fitness' : fitness_values, 'Fatigue' : fatigue_values, 'Performance' : performance_values})

result2 = result
result2['year'] = pd.DatetimeIndex(result2['date']).year
result3 = result2.loc[result2['year']>2021]

result3.plot(kind = 'line', x = 'date', y = ['Fatigue', 'Fitness','Performance'])
plt.show()

# try with one workout and look at decay variation

trimps2 = trimps[0:100]
trimps2[0] = 100
for i in range(1, len(trimps2)):
    if (i % 2) == 0:
        trimps2[i] = 0.0
    else:
        trimps2[i] = 100.0

i = 51
while i <100:
    trimps2[i] = 0
    i += 1

idx2 = pd.date_range('2020-01-01','2020-04-09')
trimps2[25] = 100
fitness_values = []
fatigue_values = []
performance_values = []
fitness = 0
fatigue = 0
r1 = 21
r2 = 12
k1 = 0.4
k2 = 0.6
for i in range(len(trimps2)):
    fitness = fitness * exp(-1 / r1) + trimps2[i]
    fatigue = fatigue * exp(-1 / r2) + trimps2[i]
    performance = fitness*k1 - fatigue*k2
    fitness_values.append(fitness)
    fatigue_values.append(fatigue)
    performance_values.append(performance)


result2 = pd.DataFrame({'date': idx2, 'Fitness' : fitness_values, 'Fatigue' : fatigue_values, 'Performance' : performance_values})

b = result2.plot(kind = 'line', x = 'date', y = ['Fatigue', 'Fitness','Performance'])



