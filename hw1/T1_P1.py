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

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])

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


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))
