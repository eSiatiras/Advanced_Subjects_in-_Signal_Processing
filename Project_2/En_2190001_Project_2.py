# -*- coding: utf-8 -*-
"""Project_2.ipynb

Ex_A
"""

import matplotlib.pyplot as plt
import scipy.io
from matplotlib import style
from scipy import linalg


style.use('ggplot')
import numpy as np

mat = scipy.io.loadmat('A.mat')
y_pts = np.asarray(mat['y'])[0:200]
y_pts = np.reshape(y_pts, -1)
plt.plot(y_pts)
plt.title("Given Data to be Predicted")
plt.axis([0, 300, -3.5, 3.5])
plt.show()

import matplotlib.pyplot as plt
from matplotlib import style
import scipy.io
import numpy as np

style.use('ggplot')


class RecursiveRegression(object):
    def __init__(self, y):
        self.y = y

    def fit(self, order=1):
        d = {}
        for j in np.arange(order, len(self.y)):
            if (j - 1 - order >= 0):
                d['y' + str(j)] = self.y[j - order:j:1][::-1]
            else:
                d['y' + str(j)] = self.y[0:order:1][::-1]

        # for key, value in d.items() :
        #   print (key, value)

        X = np.column_stack(d.values()).T
        theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), self.y[order:])

        self.theta = theta
        return self

    def plot_predictedLine(self, y_pts, start, plot):
        self.line = np.zeros(len(y_pts))
        for j in np.arange(start, len(y_pts)):
            for i in np.arange(0, len(self.theta)):
                if (j - i - 1 < start):
                    self.line[j] += self.theta[i] * y_pts[j - i - 1]
                else:
                    self.line[j] += self.theta[i] * self.line[j - i - 1]
        if (plot):
            plt.figure()
            # plt.scatter(np.arange (0, len(y_pts)), y_pts, s = 30, c = 'k')
            plt.plot(y_pts[:start], c='b', linewidth=0.5, label="y [n]")
            plt.plot(np.arange(start, len(y_pts)), self.line[start:], c='r', linewidth=1, label="y_pred [n]")
            plt.title('Recursive Fit: Order ' + str(len(self.theta)))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc='best')
            plt.show()
        return self


def PerformRegression(mat_file, start, order, plot, mse=False):
    mat = scipy.io.loadmat(mat_file)

    training_set = np.reshape(mat['y'][:start], -1)
    PR = RecursiveRegression(training_set)
    PR.fit(order=order)

    test_set = np.reshape(mat['y'], -1)
    PR.plot_predictedLine(test_set, start=start, plot=plot)

    mse_test_set = np.sum((test_set[start:] - PR.line[start:]) ** 2) * 0.01
    if (mse):
        print("MSE of Model Order: " + str(order) + " is: " + str(mse_test_set))
    return mse_test_set


line = []
min_mse = 100
order = 0
for i in np.arange(1, 90, 1):
    line = np.append(line, PerformRegression('A.mat', 200, i, plot=False))
    if (min_mse > float(np.min(line))):
        min_mse = np.min(line)
        order = i

# print (line)
# print (order)
plt.plot(np.arange(1, 90, 1), line)
plt.title('MSE')
plt.xlabel('Order')
plt.ylabel('Value')
plt.show()
PerformRegression('A.mat', 200, order, plot=True, mse=True)
PerformRegression('A.mat', 200, 20, plot=True, mse=True)
PerformRegression('A.mat', 200, 40, plot=True, mse=True)
PerformRegression('A.mat', 200, 60, plot=True, mse=True)

"""Ex_B"""

from scipy import linalg
from scipy import signal as sc
import matplotlib.pyplot as plt


class WienerFIRFilter(object):
    def __init__(self, u2, y, order):
        self.u2 = u2 - np.mean(u2)
        self.y = y - np.mean(y)
        self.lag = order

    def autocorr(self):
        c = np.correlate(self.u2, self.u2, 'full')
        mid = len(c) // 2
        # print (c[mid:mid+lag])
        acov = c[mid:mid + self.lag]
        acor = acov / acov[0]
        return (acor)

    def crosscorr(self):
        c = np.correlate(self.y, self.u2, 'full')
        mid = len(c) // 2
        # print (c[mid:mid+lag])
        ccov = c[mid:mid + self.lag]
        crosscor = ccov / ccov[0]
        return (crosscor)

    def compute_coeffs(self):
        ac = self.autocorr()
        R = linalg.toeplitz(ac[:self.lag])
        self.r = self.crosscorr()[:self.lag + 1]
        self.w_opt = linalg.inv(R).dot(self.r)
        return self.w_opt

    def predict_noise(self):
        self.u1 = sc.lfilter(w.w_opt, [1], self.u2)
        return self.u1

    def predict_signal(self):
        self.s = self.y - self.u1

    def plot_predicted_signal(self):
        plt.plot(self.s, 'b', label="Estimate s(n)")
        # plt.plot(self.y, 'r',label= "Estimate y(n)",linewidth=0.5)
        plt.title('The Predicted Signal s[n],Filter Order: ' + str(self.lag))
        # plt.legend(loc = 'best')
        plt.axis([0, 600, -2, 2])
        plt.show()

mat = scipy.io.loadmat('B.mat')

y = np.asarray(mat['y'])
u = np.asarray(mat['u'])
N = np.size(y)

u2 = np.reshape(u, -1)
y = np.reshape(y, -1)

for i in np.arange(40, 50, 1):
    w = WienerFIRFilter(u2, y, i)
    w.compute_coeffs()
    w.predict_noise()
    w.predict_signal()
    w.plot_predicted_signal()
