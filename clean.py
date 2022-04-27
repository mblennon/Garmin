from datetime import datetime
import time
import os
import json
import datetime
from collections import ChainMap
from math import exp
from matplotlib import pyplot as plt

import pandas as pd

# Get Json from folder and concat into fataframe

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

# type(runs_with_HR.startTime[0])
# runs_with_HR = runs_with_HR.sort_values(by='startTime')
# runs_with_HR2 = runs_with_HR.asfreq('D')

runs_with_HR2 = runs_with_HR.groupby(runs_with_HR.index).agg({'Trimp': sum})
runs_with_HR3 = runs_with_HR2.reindex(idx, fill_value=0)

def Banister(trimp, k1, k2, r1, r2):
    fitness = trimp * exp(-1 / r1)
    fatigue = trimp * exp(-1 / r2)
    performance = fitness * k1 - fatigue * k2
    return fitness, fatigue, performance


trimps = runs_with_HR3.Trimp.tolist()
p = []
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
    p.append(performance)

result = pd.DataFrame({'date': idx, 'Performance' : p})

result.plot(kind = 'line', x = 'date', y = 'Performance')