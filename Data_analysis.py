# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 08:56:33 2021

@author: alexh
"""

# Importing necessary modules

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics as st

# Free damped decay file names

free_damped = []
for i in range(1, 9):
    free_damped.append(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\Free damped decay data\X2sv0000" + f"{i}.txt")
#print(free_damped)

# Forced response file names

forced_response = []
for i in range(1, 10):
    forced_response.append(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\Forced response data\Xxsv0000" + f"{i}.txt")
#forced_response.append(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\Forced response data\Xxsv00010.txt")
#print(forced_response)
    
#%%

""" 

    Defining functions to read the free damped decay and the forced response data
    from fixed width format into pandas data frames, and functions to save 
    plots of that data as images.

"""

def read_free(data):
    table = pd.read_fwf(data, widths=[20, 20], encoding='cp1252')
    table = table.drop(table.iloc[0:6].index)
    table = table.set_axis(["time_[s]", "acceleration_[m/s^2]"], axis=1)
    table = table.astype("float64")
    table = table.set_index("time_[s]")
    return table

def read_forced(data):
    table = pd.read_fwf(data, widths=[20, 20, 20], encoding='cp1252')
    table = table.drop(table.iloc[0:8].index)
    table = table.set_axis(["time_[s]", "Force_[N]", "acceleration_[m/s^2]"], axis=1)
    table = table.astype("float64")
    table = table.set_index("time_[s]")
    return table


def free_plotter(data_set, n):
    fig, ax = plt.subplots(dpi=1000)
    ax.plot(data_set.index.values, data_set["acceleration_[m/s^2]"].values, linewidth=0.1)
    ax.set_title("Free damped decay")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration $[m/s^2]$")
    ax.grid(linewidth=0.2)
    plt.savefig(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\free_damped_decay_0" + f"{n + 1}")
    
def forced_plotter(data_set, n):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=1000)
    ax1.plot(data_set.index.values, data_set["Force_[N]"].values, linewidth=0.1)
    ax2.plot(data_set.index.values, data_set["acceleration_[m/s^2]"].values, linewidth=0.1)
    ax1.set_title("Forced Response")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("Force [N]")
    ax1.grid(linewidth=0.2)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("acceleration $[m/s^2]$")
    ax2.grid(linewidth=0.2)
    plt.savefig(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\forced_vibration_0" + f"{n + 1}")

#%%

"""Free damped oscillation data analysis"""

# reading in the all of the data and saving the graphs into images for selection of best data:
    
for i in range(len(free_damped)):
    free_plotter(read_free(free_damped[i]), i)
    
#%%

# Reading all data into a list of dataframes

free_data = []
for i in range(len(free_damped)):
    free_data.append(read_free(free_damped[i]))


#%%


"""

Runs 6, 7 and 8 look the cleanest. Now we need to calculate the damping
ratio and the natural frequency.

To calculate the frequency we need to count the number of cycles in a given 
time period, and divide that by the time period. The time period
should be long to get a high number of repeats to improve accuracy. we can 
also average across the three runs to get the overall natural frequency.

As for the damping ratio, we need to pick the maxima on two consecutive cycles
and then compute the natural logarithm of their fraction, then divide that 
by 2pi.

We could do this by making functions or by looking at the data directly and 
computing. The latter is probably eventually the quicker option, but annoying
to repeat in case of mistakes, plus this is good practice. 

"""

# In a cycle, the function passes the equilibrium mark twice,
# or more simply, if it moves from negative to positive once

eq_1 = free_data[5]["acceleration_[m/s^2]"].mean()
eq_2 = free_data[6]["acceleration_[m/s^2]"].mean()
eq_3 = free_data[7]["acceleration_[m/s^2]"].mean()
    
    
def passed_zero_up(couple, eq):
    if couple.iloc[0] <= eq and couple.iloc[1] >= eq:
        return 1
    else:
        return 0
 
def frequency_calculator(data_slice, equilibrium):
    cycle_check = data_slice.rolling(window=2).apply(lambda x: passed_zero_up(x, equilibrium))
    #import pdb; pdb.set_trace()
    cycles = cycle_check.sum()
    time_period = (data_slice.index[-1] - data_slice.index[0]) / cycles
    frequency = 1 / time_period
    return frequency

#%%

# Starting the real data analysis: natural frequency calculation

freq_1 = frequency_calculator(free_data[5]["acceleration_[m/s^2]"].loc[3:15], eq_1)
freq_2 = frequency_calculator(free_data[6]["acceleration_[m/s^2]"].loc[3:15], eq_2)
freq_3 = frequency_calculator(free_data[7]["acceleration_[m/s^2]"].loc[6:20], eq_3)

freq_av = sum([freq_1, freq_2, freq_3]) / 3
angular_damped_frequency = freq_av * 2 * math.pi

print(freq_av)
print(angular_damped_frequency)
# Notice the good sign that this looks very close to the frequency at which
# resonance was measured in the lab for the forced vibration tests


#%%

# Damping ratio calculation

"""
    The damping ratio zeta is calculated by (1/2pi)*ln(x_{t_m})/x(t_{m+1}))
    This is that task that probably really has the highest justification for
    doing the data analysis all in python, because otherwise it would be quite
    annoying to look up in a table
"""

# Function to calculate a damping ratio

def get_damping_ratio(data_slice, equilibrium):
    cycle_check = data_slice.rolling(window=2).apply(lambda x: passed_zero_up(x, equilibrium))
    
    repeat_counter = 0
    for measure in cycle_check.iterrows():
        if measure[1].iloc[0] == 1:
            if repeat_counter == 0:
                repeat_counter += 1
                first_time = measure[0]
            elif repeat_counter == 1:
                second_time = measure[0]
                repeat_counter += 1
            elif repeat_counter == 2:
                third_time = measure[0]
                break
    
    max_acc_1 = data_slice[first_time:second_time].max()
    max_acc_2 = data_slice[second_time:third_time].max()
    #import pdb; pdb.set_trace()
    damping_ratio = (1 / (2 * math.pi)) * math.log(max_acc_1 / max_acc_2)
    
    return damping_ratio


ratios_1 = []
ratios_2 = []
ratios_3 = []
step = 0.5
for i in np.arange(3, 13, step):
    ratios_1.append(get_damping_ratio(free_data[5].loc[i:i+1], eq_1))
for i in np.arange(3, 15, step):
    ratios_2.append(get_damping_ratio(free_data[6].loc[i:i+1], eq_2))
for i in np.arange(6, 20, step):
    ratios_3.append(get_damping_ratio(free_data[7].loc[i:i+1], eq_3))
    
all_ratios = [ratios_1, ratios_2, ratios_3]

#%%

ratios_1_avg = st.mean(ratios_1)
ratios_2_avg = st.mean(ratios_2)
ratios_3_avg = st.mean(ratios_3)

half_ratios_1 = ratios_1[0:int((len(ratios_1)/2))]
half_ratios_1_avg = st.mean(half_ratios_1)

half_ratios_2 = ratios_1[0:int((len(ratios_2)/2))]
half_ratios_2_avg = st.mean(half_ratios_2)

half_ratios_3 = ratios_3[0:int((len(ratios_3)/2))]
half_ratios_3_avg = st.mean(half_ratios_3)

quarter_ratios_1 = ratios_1[0:int((len(ratios_1)/4))]
quarter_ratios_1_avg = st.mean(quarter_ratios_1)

quarter_ratios_2 = ratios_1[0:int((len(ratios_2)/4))]
quarter_ratios_2_avg = st.mean(quarter_ratios_2)

quarter_ratios_3 = ratios_3[0:int((len(ratios_3)/4))]
quarter_ratios_3_avg = st.mean(quarter_ratios_3)

print(ratios_1_avg, ratios_2_avg, ratios_3_avg, half_ratios_1_avg, half_ratios_2_avg, half_ratios_3_avg, quarter_ratios_1_avg, quarter_ratios_2_avg, quarter_ratios_3_avg, sep="\n")

quarter_overall_damping_ratio = st.mean([quarter_ratios_1_avg, quarter_ratios_2_avg, quarter_ratios_3_avg])
half_overall_damping_ratio = st.mean([half_ratios_1_avg, half_ratios_2_avg, half_ratios_3_avg])
overall_damping_ratio = st.mean([ratios_1_avg, ratios_2_avg, ratios_3_avg])

print("\n", quarter_overall_damping_ratio, half_overall_damping_ratio, overall_damping_ratio, sep="\n")

#%%

# Now for the FRF data

# Reading all data from forced_response

forced_data = []
for i in range(len(forced_response)):
    forced_data.append(read_forced(forced_response[i]))


#%%

# Reading all of the data into saved images


for i in range(len(forced_data)):
    forced_plotter(read_forced(forced_response[i]), i)

#%%

# Calculating averages of force and acceleration


def get_max_in_cycle(data_slice, equilibrium):
    cycle_check = data_slice.rolling(window=2).apply(lambda x: passed_zero_up(x, equilibrium))
    
    first_time = True
    for measure in cycle_check.items():
        if measure[1] == 1:
            if first_time:
                first_time = False
                time_1 = measure[0]
            else:
                time_2 = measure[0]
                break
    
    maxi = data_slice[time_1:time_2].max()
    
    return maxi
    


def get_avg_max(data_set, eq):
    numbers = []
    for i in range(1, int(data_set.index.values[-1])):
        numbers.append(get_max_in_cycle(data_set.loc[i:i+1], eq))
    return st.mean(numbers)

# Producing table of Force and acceleration amplitude vs frequency

F_offsets = []
a_offsets = []

for i in range(len(forced_data)):
    F_offsets.append(forced_data[i]["Force_[N]"].mean())
    a_offsets.append(forced_data[i]["acceleration_[m/s^2]"].mean())
                                    

F_avgs = []
a_avgs = []
frequencies = []

for i in range(len(forced_data)):
    F_avgs.append(get_avg_max(forced_data[i]["Force_[N]"], F_offsets[i]))
    a_avgs.append(get_avg_max(forced_data[i]["acceleration_[m/s^2]"], a_offsets[i]))
    frequencies.append(frequency_calculator(forced_data[i]["Force_[N]"], F_offsets[i]))


Data = pd.DataFrame({"Frequency": frequencies, "Force_amplitude": F_avgs, "acceleration_amplitude": a_avgs})

Data["Angular_frequency"] = Data["Frequency"] * (2 * math.pi)


Data["Displacement_amplitude"] = Data["acceleration_amplitude"] / (Data["Angular_frequency"] ** 2)

Data["x/F"] = Data["Displacement_amplitude"] / Data["Force_amplitude"]

Data["Accelerance"] = Data["acceleration_amplitude"] / Data["Force_amplitude"]

Data["FRF"] = Data["Displacement_amplitude"] / Data["Force_amplitude"]


#%%

"""
    Now the most difficult bit, calculating the phase difference between the
    Force and the accelaration graphs
"""

def passed_zero_up_ez(couple):
    if couple.iloc[0] <= 0 and couple.iloc[1] >= 0:
        return 1
    else:
        return 0

def get_phase_difference(data_slice, frequency):
    cycle_check = data_slice["Force_[N]"].rolling(window=2).apply(passed_zero_up_ez)
    
    first_time = True
    for measure in cycle_check.items():
        if measure[1] == 1:
            if first_time:
                first_time = False
                time_1 = measure[0]
            else:
                time_2 = measure[0]
                break
    #import pdb; pdb.set_trace()
    max_time_1 = data_slice.loc[time_1:time_2, "Force_[N]"].idxmax()
    max_time_2 = data_slice.loc[max_time_1 - (0.5 * (1 / frequency)):(max_time_1 + (0.5 *(1 / frequency))), "acceleration_[m/s^2]"].idxmax()
    
    phase_difference = 2 * math.pi * (max_time_1 - max_time_2) * frequency
    
    return phase_difference

def pos_neg(x):
    if x >= 0:
        return x - math.pi
    elif x < 0:
        return x + math.pi


def get_phase_difference_average(data_slice, frequency):
    phase_difference_list = []
    for i in range(1, int(data_slice.index[-2])):
        phase_difference_list.append(get_phase_difference(data_slice[i:(i + 1)], frequency))
    final = []
    
    for diff in phase_difference_list:
        final.append(pos_neg(diff))
    #import pdb; pdb.set_trace()
    return st.mean(final)


phase_differences = []
for i in range(len(forced_data)):
    phase_differences.append(get_phase_difference_average(forced_data[i], Data["Frequency"].iloc[i]))

Data["Phase_difference"] = phase_differences


#%%

Data = Data.sort_values("Frequency")

Data["Frequency"] = Data["Frequency"].round(2)
Data["Force_amplitude"] = Data["Force_amplitude"].round(2)
Data["acceleration_amplitude"] = Data["acceleration_amplitude"].round(3)
Data["Displacement_amplitude"] = Data["Displacement_amplitude"].round(6)
Data["Accelerance"] = Data["Accelerance"].round(3)
Data["FRF"] = Data["FRF"].round(6)
Data["Phase_difference"] = Data["Phase_difference"].round(3)
Data["Angular_frequency"] = Data["Angular_frequency"].round(3)
Data["x/F"] = Data["x/F"].round(3)

Data.to_csv("C:/Users/alexh/OneDrive/Documents/Mechanical Engineering Year 2/PEN/Dynamics Lab/forced_response_data.csv")


#%%

import scipy.optimize as opt

def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=500)
ax1.scatter(Data["Angular_frequency"].values, Data["x/F"].values, s=4, linewidth=1)
ax2.scatter(Data["Angular_frequency"], Data["Phase_difference"].values, s=4, linewidth=1)

z = np.polyfit(Data["Angular_frequency"], Data["x/F"], 8)
p = np.poly1d(z)
ax1.plot(Data["Angular_frequency"], p(Data["Angular_frequency"]))

ax2.plot(Data["Angular_frequency"], Data["Phase_difference"].values)

ax1.set_title("Frequency Response Function - Bode Plot")
ax1.set_ylabel(r"$x_0/F$ [m/N]")
ax1.grid(linewidth=0.2)
ax2.set_xlabel("Excitation frequency, $\omega_f$ [rads$^{-1}$]")
ax2.set_ylabel("Phase difference, $\phi$ [rad]")
ax2.grid(linewidth=0.2)
plt.savefig(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\Bode_plot.png")

#%%

import scipy.optimize as opt

def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=500)
ax1.scatter(Data["Angular_frequency"].values, Data["Accelerance"].values, s=4, linewidth=1)
ax2.scatter(Data["Angular_frequency"], Data["Phase_difference"].values, s=4, linewidth=1)

z = np.polyfit(Data["Angular_frequency"], Data["Accelerance"], 8)
p = np.poly1d(z)
ax1.plot(Data["Angular_frequency"], p(Data["Angular_frequency"]))

ax2.plot(Data["Angular_frequency"], Data["Phase_difference"].values)

ax1.set_title("Frequency Response Function - Bode Plot")
ax1.set_ylabel(r"Accelerance, $\ddot{x}/F$ [m/s$^2$/N]")
ax1.grid(linewidth=0.2)
ax2.set_xlabel("Excitation frequency, $\omega_f$ [rads$^{-1}$]")
ax2.set_ylabel("Phase difference, $\phi$ [rad]")
ax2.grid(linewidth=0.2)
plt.savefig(r"C:\Users\alexh\OneDrive\Documents\Mechanical Engineering Year 2\PEN\Dynamics Lab\Bode_plot.png")
