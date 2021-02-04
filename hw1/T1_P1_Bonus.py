###################################### Bonus Gradient Descent (5) #############################
import math
import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

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

    for n in range(len(data)):
        if n != i:
            x_n = [data[n][0], data[n][1]]
            y_n = data[n][2]
            numerator_sum += kernel(x_n, x_i, W)*y_n

    denominator_sum = 0

    for n in range(len(data)):
        if n != i:
            x_n = [data[n][0], data[n][1]]
            y_n = data[n][2]
            denominator_sum += kernel(x_n, x_i, W)

    return numerator_sum/denominator_sum

def l2_residual(x_i, y_i, i, W):
    return (y_i - regressor(x_i, i, W))**2

def compute_loss(W):

    loss = 0

    for i in range(len(data)):
        x_i = [data[i][0], data[i][1]]
        y_i = data[i][2]
        loss += l2_residual(x_i, y_i, i, W)


    return loss

def e_ni(w_11, w_12, w_22, a_ni, b_ni):

    return math.exp(-((a_ni**2)*w_11 + 2*a_ni*b_ni*w_12 + (b_ni**2)*w_22))

def C_i(w_11, w_12, w_22, x_i, y_i, i):

    sum_e_ni = 0
    sum_e_ni_y_n = 0

    for n in range(len(data)):

        x_n = [data[n][0], data[n][1]]
        y_n = data[n][2]

        a_ni = x_n[0] - x_i[0]
        b_ni = x_n[1] - x_i[1]

        if i != n:

            e_ni_val = e_ni(w_11, w_12, w_22, a_ni, b_ni)
            sum_e_ni += e_ni_val
            sum_e_ni_y_n += e_ni_val * y_n

    return y_i - (sum_e_ni_y_n / sum_e_ni)


def gradient_loss(w_11, w_12, w_22):

    result_w_11 = 0
    result_w_12 = 0
    result_w_22 = 0

    for i in range(len(data)):
        x_i = [data[i][0], data[i][1]]
        y_i = data[i][2]

        sum_e_ni = 0
        sum_e_ni_y_n = 0
        # with respect to w_11
        sum_minus_a_ni_squared_e_ni = 0
        sum_minus_a_ni_squared_e_ni_y_n = 0
        # with respect to w_22
        sum_minus_b_ni_squared_e_ni = 0
        sum_minus_b_ni_squared_e_ni_y_n = 0
        # with respect to w_12
        sum_minus_2ab_ni_e_ni = 0
        sum_minus_2ab_ni_e_ni_y_n = 0

        for n in range(len(data)):
            x_n = [data[n][0], data[n][1]]
            y_n = data[n][2]

            a_ni = x_n[0] - x_i[0]
            b_ni = x_n[1] - x_i[1]

            if i != n:
                e_ni_val = e_ni(w_11, w_12, w_22, a_ni, b_ni)

                sum_e_ni += e_ni_val
                sum_e_ni_y_n += e_ni_val*y_n
                # with respect to w_11
                sum_minus_a_ni_squared_e_ni += (-1*a_ni**2)*e_ni_val
                sum_minus_a_ni_squared_e_ni_y_n += (-1*a_ni**2)*e_ni_val*y_n
                # with respect to w_22
                sum_minus_b_ni_squared_e_ni += (-1*b_ni**2)*e_ni_val
                sum_minus_b_ni_squared_e_ni_y_n += (-1*b_ni**2)*e_ni_val*y_n
                # with respect to w_12
                sum_minus_2ab_ni_e_ni += (-2*a_ni*b_ni)*e_ni_val
                sum_minus_2ab_ni_e_ni_y_n += (-2*a_ni*b_ni)*e_ni_val*y_n

        C_i_val = C_i(w_11, w_12, w_22, x_i, y_i, i)
        val_i = ((sum_e_ni*sum_minus_a_ni_squared_e_ni_y_n) - (sum_e_ni_y_n*sum_minus_a_ni_squared_e_ni))*C_i_val
        val_i = val_i/(sum_e_ni**2)
        #print(val_i)
        result_w_11 += val_i
        val_i = ((sum_e_ni*sum_minus_b_ni_squared_e_ni_y_n) - (sum_e_ni_y_n*sum_minus_b_ni_squared_e_ni))*C_i_val
        val_i = val_i/(sum_e_ni**2)
        #print(val_i)
        result_w_22 += val_i
        val_i = ((sum_e_ni*sum_minus_2ab_ni_e_ni_y_n) - (sum_e_ni_y_n*sum_minus_2ab_ni_e_ni))*C_i_val
        val_i = val_i/(sum_e_ni**2)
        #print(val_i)
        result_w_12 += val_i

    return [-result_w_11, -result_w_12, -result_w_22]

import random

def gradient_descent(n):
    l_const = 1
    #w_0 = [random.choice([i/10 for i in range(10)]), random.choice([i/10 for i in range(10)]), random.choice([i/10 for i in range(10)])]
    w_0 = [1,0,1]
    w_n = w_0
    w_n = np.array([w_n]).reshape(3,1)
    for i in range(n):
        w_n = w_n - l_const*np.array([gradient_loss(w_n[0][0], w_n[1][0], w_n[2][0])]).reshape(3,1)

    return w_n

w = gradient_descent(100)
w_arr = alpha * np.array([[w[0][0], w[1][0]], [w[1][0], w[2][0]]])
print(w_arr)
print("L2 Loss: {}".format(compute_loss(w_arr)))