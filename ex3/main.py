import numpy as np
import matplotlib.pyplot as plt
import os

training_data = np.loadtxt(os.path.join('data_ml', 'training_data.txt'))
validation_data = np.loadtxt(os.path.join('data_ml', 'validation_data.txt'))

tx = training_data[0]
ty = training_data[1]
vx = validation_data[0]
vy = validation_data[1]


def LLS(x, y, feature_func):
    f_matrix = feature_func(x)
    return np.linalg.inv(f_matrix.T  @ f_matrix) @ f_matrix.T @ y


def sin_feature(x, n):
    return np.sin(2 ** np.mgrid[0:n].reshape(-1, 1) * x).T


def rmse(y, yp):
    return np.sqrt(np.sum((y - yp) ** 2) / y.shape[0])


def exp_sq_kernel(xi, xj, sigma=.15):
    return np.exp(-np.abs(xi - xj) ** 2 / (sigma ** 2))


F = True
# c)
if 'C' in locals():
    sin_fs = [lambda x, i=i: sin_feature(x, i) for i in [2, 3, 9]]
    thetas = [LLS(tx, ty, f) for f in sin_fs]

    x = np.linspace(0, 6, 601)
    yps = [sin_fs[i](x) @ theta for i, theta in enumerate(thetas)]

    for i, y in enumerate(yps):
        plt.plot(x, y, label=f"{[2, 3, 9][i]}")

    plt.legend()
    plt.show()

# d)
elif 'D' in locals():
    indexes = np.linspace(1, 9, 9, dtype=int)
    sin_fs = [lambda x, i=i: sin_feature(x, i) for i in indexes]
    thetas = [LLS(tx, ty, f) for f in sin_fs]

    yps = [sin_fs[i](tx) @ theta for i, theta in enumerate(thetas)]

    for i, yp in enumerate(yps):
        plt.plot(tx, yp, label=f"{indexes[i]}")

    plt.plot(tx, ty, label="Original data")
    plt.legend()
    plt.show()

    plt.bar(indexes, np.array([rmse(yp, ty) for yp in yps]))

    plt.show()

# e)
elif 'E' in locals():
    indexes = np.linspace(1, 9, 9, dtype=int)
    sin_fs = [lambda x, i=i: sin_feature(x, i) for i in indexes]
    thetas = [LLS(tx, ty, f) for f in sin_fs]

    yps = [sin_fs[i](vx) @ theta for i, theta in enumerate(thetas)]

    for i, yp in enumerate(yps):
        plt.plot(vx, yp, label=f"{indexes[i]}")

    plt.plot(vx, ty, label="Original data")
    plt.legend()
    plt.show()

    plt.bar(indexes, np.array([rmse(yp, vy) for yp in yps]))
    print(np.array([rmse(yp, vy) for yp in yps]))

    plt.show()

# f)
elif 'F' in locals():
    indexes = np.linspace(1, 9, 9, dtype=int)
    sin_fs = [lambda x, i=i: sin_feature(x, i) for i in indexes]

    r = np.zeros((2, 0))
    for f in sin_fs:
        rmses = np.zeros((0,))
        for leave_out in range(tx.shape[0]):
            left_out = tx[leave_out]
            test_set_x = np.hstack((tx[:leave_out], tx[leave_out+1:]))
            test_set_y = np.hstack((ty[:leave_out], ty[leave_out+1:]))
            theta = LLS(test_set_x, test_set_y, f)
            rmses = np.append(rmses, rmse(f(left_out)@theta, ty[leave_out]))
        r = np.hstack((r, np.array([[np.mean(rmses)], [np.var(rmses)]])))

    plt.title("RMSE LOO")
    plt.plot(indexes, r[0], color='blue')
    plt.fill_between(indexes, r[0] - np.sqrt(r[1]), r[0] + np.sqrt(r[1]),
                     color='orange')
    plt.xlabel("Sine order")
    plt.ylabel(u"Mean RMSE \u00B1 std")
    plt.legend()
    plt.show()


elif 'H' in locals():
    tx_ = tx.reshape(-1, 1)
    K = exp_sq_kernel(tx_, tx_.T)
    xs = np.linspace(0, 6, 601).reshape(-1, 1)
    k = exp_sq_kernel(vx.reshape(-1, 1), tx_.T)

    print(rmse((k @ np.linalg.inv(K) @ ty).reshape(-1), ty.reshape(-1)))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(vx, k @ np.linalg.inv(K) @ ty, label='Predicted output')
    plt.scatter(vx, vy, color='orange', label='Validation data')
    plt.title('Predicted output on validation input')
    plt.legend()
    plt.show()
