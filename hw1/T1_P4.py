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
plt.close()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20

    result = []
    if part == 'a':
        for val in xx:
            arr = [val**e for e in range(1,6)]
            arr.insert(0, 1)
            result.append(arr)
    elif part == 'b':
        for val in xx:
            arr = [math.exp((-(val - e)**2)/25) for e in range(1960,2015,5)]
            arr.insert(0, 1)
            result.append(arr)
    elif part == 'c':
        for val in xx:
            arr = [math.cos(val/e) for e in range(1,6)]
            arr.insert(0, 1)
            result.append(arr)
    else:
        for val in xx:
            arr = [math.cos(val/e) for e in range(1,26)]
            arr.insert(0, 1)
            result.append(arr)
    return np.array(result)
        
        
    return None

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

def find_residual_values(X, Y, w):
    return np.sum(np.square(Y - np.dot(X,w)))

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



print(residuals)



# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
#grid_Yhat  = np.dot(grid_X.T, w)

X1_basis = make_basis(years, part='a')
X2_basis = make_basis(years, part='b')
X3_basis = make_basis(years, part='c')
X4_basis = make_basis(years, part='c')

grid_Yhat1  = np.dot(make_basis(grid_years, part='a'), w1)
grid_Yhat2  = np.dot(make_basis(grid_years, part='b'), w2)
grid_Yhat3  = np.dot(make_basis(grid_years, part='c'), w3)
grid_Yhat4  = np.dot(make_basis(grid_years, part='d'), w4)


# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
plt.title("A. Quintic Polynomial Basis")
plt.scatter(years, republican_counts, label='L2: ' + str(residuals[0]))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat1, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(1) + '.png')
plt.close()
plt.title("B. Expo Fraction Transformation")
plt.scatter(years, republican_counts, label='L2: ' + str(residuals[1]))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat2, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(2) + '.png')
plt.close()
plt.title("C. cos(x / j), j=[1,5]")
plt.scatter(years, republican_counts, label='L2: ' + str(residuals[2]))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat3, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(3) + '.png')
plt.close()
plt.title("D. cos(x / j), j=[1,25]")
plt.scatter(years, republican_counts, label='L2: ' + str(residuals[3]))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat4, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(4) + '.png')
plt.close()

sunspot_counts_1985 = sunspot_counts[years<last_year] 
republican_counts_1985 = republican_counts[years<last_year]

X1_basis = make_basis(sunspot_counts_1985, part='a', is_years=False)
X3_basis = make_basis(sunspot_counts_1985, part='c', is_years=False)
X4_basis = make_basis(sunspot_counts_1985, part='d', is_years=False)

w1 = find_weights(X1_basis, republican_counts_1985)
w3 = find_weights(X3_basis, republican_counts_1985)
w4 = find_weights(X4_basis, republican_counts_1985)

residuals = []

residuals.append(find_residual_values(X1_basis, republican_counts_1985, w1))
residuals.append(find_residual_values(X3_basis, republican_counts_1985, w3))
residuals.append(find_residual_values(X4_basis, republican_counts_1985, w4))

grid_sunspots = np.linspace(0, 155, 200)

X1_basis = make_basis(grid_sunspots , part='a', is_years=False)
X3_basis = make_basis(grid_sunspots , part='c', is_years=False)
X4_basis = make_basis(grid_sunspots , part='d', is_years=False)

grid_Yhat1  = np.dot(X1_basis, w1)
grid_Yhat3  = np.dot(X3_basis, w3)
grid_Yhat4  = np.dot(X4_basis, w4)

plt.title("A. Quintic Polynomial Basis")
plt.scatter(sunspot_counts_1985, republican_counts_1985, label='L2: ' + str(residuals[0]))
plt.plot(sunspot_counts_1985, republican_counts_1985, 'o', grid_sunspots, grid_Yhat1, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(5) + '.png')
plt.close()
plt.title("C. cos(x / j), j=[1,5]")
plt.scatter(sunspot_counts_1985, republican_counts_1985, label='L2: ' + str(residuals[1]))
plt.plot(sunspot_counts_1985, republican_counts_1985, 'o', grid_sunspots, grid_Yhat3, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(7) + '.png')
plt.close()
plt.title("D. cos(x / j), j=[1,25]")
plt.scatter(sunspot_counts_1985, republican_counts_1985, label='L2: ' + str(residuals[2]))
plt.plot(sunspot_counts_1985, republican_counts_1985, 'o', grid_sunspots, grid_Yhat4, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig('r' + str(8) + '.png')
