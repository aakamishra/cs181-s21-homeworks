#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

def kernel(x_n, x_i, W):
    x_n = np.array([x_n]).reshape(2,1)
    x_i = np.array([x_i]).reshape(2,1)

    x_n_sub_x_i = x_n - x_i
    x_n_sub_x_i_T = np.transpose(x_n_sub_x_i)
    x_n_sub_x_i_T_dot_W = np.dot(x_n_sub_x_i_T, W)
    product = np.dot(x_n_sub_x_i_T_dot_W, x_n_sub_x_i)
    expo = np.exp(-product[0][0])

    return expo

def regressor(x_i, i, W):
    numerator_sum = 0
    N = len(X_df.values)
    for n in range(N):
        if n != i:
            x_n = X_df.values[n]
            y_n = y_df.values[n]
            numerator_sum += kernel(x_n, x_i, W)*y_n

    denominator_sum = 0

    for n in range(N):
        if n != i:
            x_n = X_df.values[n]
            y_n = y_df.values[n]
            denominator_sum += kernel(x_n, x_i, W)

    return numerator_sum/denominator_sum

def predict_kernel(alpha=0.1):
    W = alpha * np.array([[1., 0.], [0., 1.]])
    i = 0
    pred_y = []
    for x_i in X_df.values:
        pred_y.append(regressor(x_i, i, W))
        i += 1
    return pred_y

def compare(dist1, dist2):
    if dist1[0] < dist2[0]:
        return -1
    elif dist1[0] > dist2[0]:
        return 1
    else:
        return 0

def predict_knn(k=1):
    pred_y = []
    W = 1 * np.array([[1., 0.], [0., 1.]])

    
    for i in range(len(X_df.values)):
        dist_list = []
        for j in range(len(X_df.values)):
            if i != j:
                x_n = np.array([X_df.values[j]]).reshape(2,1)
                x_i = np.array([X_df.values[i]]).reshape(2,1)

                x_n_sub_x_i = x_n - x_i
                x_n_sub_x_i_T = np.transpose(x_n_sub_x_i)
                x_n_sub_x_i_T_dot_W = np.dot(x_n_sub_x_i_T, W)
                product = np.dot(x_n_sub_x_i_T_dot_W, x_n_sub_x_i)
                dist_list.append((product[0][0], j))

        dist_list = sorted(dist_list, key=lambda x: x[0])

        suma = 0
        for i in range(k):
            index = dist_list[i][1]
            suma += y_df.values[index]
        pred_y.append(suma/k)

        i += 1

    print("k: {}, pred: {}".format(k, pred_y))
    return pred_y

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b', label='L2: ' + str(sum((y - y_pred) ** 2)))
    plt.legend()
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b', label='L2: ' + str(sum((y - y_pred) ** 2)))
    plt.legend()
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
