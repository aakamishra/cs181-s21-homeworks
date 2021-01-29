#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
def make_basis(xx,part='a'):
    if part == 'a':
        result = np.array([])
        for val in xx:
            arr = [val**e for e in range(1,6)]
            arr.insert(0, 1)
            np.append(result, [arr], axis=0)
        return result
    elif part == 'b':
        result = np.array([])
        for val in xx:
            arr = [math.exp((-(val - e)**2)/25) for e in range(1960,2015,5)]
            arr.insert(0, 1)
            np.append(result, [arr], axis=0)
        return result
    elif part == 'c':
        result = np.array([])
        for val in xx:
            arr = [math.cos(val/e) for e in range(1,6)]
            arr.insert(0, 1)
            np.append(result, [arr], axis=0)
        return result
    else:
        result = np.array([])
        for val in xx:
            arr = [math.cos(val/e) for e in range(1,26)]
            arr.insert(0, 1)
            np.append(result, [arr], axis=0)
        return result



# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
    return w

def find_residual_values(X, Y, w):
    return np.sum(np.square(Y - np.dot(w,X)))

X1_basis = make_basis(years, part='a')
X2_basis = make_basis(years, part='b')
X3_basis = make_basis(years, part='c')
X4_basis = make_basis(years, part='d')

w1 = find_weights(X1_basis, republican_counts)
w2 = find_weights(X2_basis, republican_counts)
w3 = find_weights(X3_basis, republican_counts)
w4 = find_weights(X4_basis, republican_counts)

residuals = []

residuals.append(find_residual_values(X1_basis, republican_counts, w1))
residuals.append(find_residual_values(X2_basis, republican_counts, w2))
residuals.append(find_residual_values(X3_basis, republican_counts, w3))
residuals.append(find_residual_values(X4_basis, republican_counts, w4))



# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()

