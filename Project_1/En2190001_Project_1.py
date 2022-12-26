# !pip install spectrum
# !pip install scipy
# !pip install matplotlib
# !pip install numpy
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import spectrum as sp
from scipy import signal, linalg

"""Ex_A_1"""

N = 1000
ni = np.arange(N)

time_data = 4.5 * np.cos(0.4 * np.pi * ni) + 2 * np.cos(0.5 * np.pi * ni) + signal.unit_impulse(N)

frequency = np.fft.fftfreq(N)
freq_data = np.fft.fft(time_data)
y = 1 / N * np.abs(freq_data)

# Plotting the FFT spectrum
plt.plot(frequency, y)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

"""EX_A_2"""

N = 64

ni = np.arange(N)
fi1 = np.random.uniform(-np.pi, np.pi, 1)
fi2 = np.random.uniform(-np.pi, np.pi, 1)

noise = np.random.normal(loc=0, scale=1, size=N)

x = 3 * np.sin(0.4 * np.pi * ni + fi1) + 2 * np.sin(0.5 * np.pi * ni + fi2)
x += noise

plt.plot((x[:64]))
plt.title('signal')
plt.ylabel('Signal strength')
plt.xlabel('Sample number')
plt.show()


class YW(object):
    def __init__(self, X):
        self.X = X - np.mean(X)

    def autocorr(self, lag=16):
        c = np.correlate(self.X, self.X, 'full')
        mid = len(c) // 2
        # print (c[mid:mid+lag])
        acov = c[mid:mid + lag]
        acor = acov / acov[0]
        return (acor)

    def compute_coeffs(self, p=15):
        self.p = p
        #
        ac = self.autocorr(p + 1)
        R = linalg.toeplitz(ac[:p])
        r = ac[1:p + 1]
        self.a = linalg.inv(R).dot(r)
        return self.a

    def power_spectrum(self):
        a = np.concatenate([np.ones(1), -self.a])
        w, h = signal.freqz(1, a)
        h_db = 10 * np.log10(2 * (np.abs(h) / len(h)))
        plt.plot(w / np.pi, h_db, label='AR_Model_of_Order_%s' % self.p)
        plt.xlabel(r'Normalized Frequency ($\times \pi$rad/sample)')
        plt.ylabel(r'Power/frequency (dB/rad/sample)')
        plt.title(r'Yule-Walker Spectral Density Estimate')
        plt.legend()


# Our goal is to estimate the parameters of the model that best fits the data of the discrete random process
# With an AR model of order 15 we can assume that the model is starting to be similar with our data.
p = sp.Periodogram(x, sampling=2)
p()
p.plot(label='PSD of model output')
ar15 = YW(x)
ar15.compute_coeffs(15)
ar15.power_spectrum()
print (ar15.a)
print (sp.aryule(x, 15)[0])
plt.show()

"""Ex_A_3"""

AR_Model_Order = 40
AR, P, k = sp.aryule(x, AR_Model_Order)

# Let's use the Yule-Walker method to fit an AR(40) model to the process and plot
# the reflection coefficients, whose negative is the partial autocorrelation sequence
pacf = -k

plt.stem(pacf)
plt.axis([-0.05, AR_Model_Order + 0.5, -1, 1])
plt.title('Reflection Coefficients by Lag')
plt.xlabel('Lag')
plt.ylabel('Reflection Coefficent')
plt.show()

# We define a large-sample 95% confidence intervals.
conf = 1.96 / np.sqrt(N)
print ("The Confidence interval for the given number of samples is: ", conf)
for i in range(0, pacf.size):
    if pacf[i] > 0:
        if pacf[i] > conf:
            last_coeff = i
            print ("Reflection Coefficient ", i, " is outsize confidence intervals")
    if pacf[i] < 0:
        if pacf[i] < -conf:
            last_coeff = i
            print ("Reflection Coefficient ", i, " is outsize confidence intervals")

AR_Model_Order = last_coeff
AR, P, k = sp.aryule(x, AR_Model_Order)
print ("The AR model coefficients are: ", AR)

y = np.zeros(N)
for n in range(0, N):
    for k in range(0, AR_Model_Order):
        if n - k >= 0:
            y[n] = - AR[k] * y[n - k]
        y[n] += noise[n]
print (y)

"""Ex_A_4"""

y = signal.lfilter([1], list(AR), np.transpose(noise.reshape(-1, 1)))

p = sp.pyule(y[0], AR_Model_Order)  # Calculation of PSD
p()
p.plot(label='PSD estimate of y using Yule-Walker AR(' + str(AR_Model_Order) + ')')
############### Model from Question 3 ##################
PSD = sp.arma2psd(AR, NFFT=512)
PSD = PSD[len(PSD):len(PSD) // 2:-1]
plt.plot(np.linspace(0, 1, len(PSD)), 10 * np.log10(abs(PSD) * 2. / (2. * np.pi)),
     label='Estimate of x using Yule-Walker AR(8)')
############### Model from Question 1 ##################
p = sp.Periodogram(x, sampling=2)
p()
p.plot(label='PSD of model Output')
############### Model from Question 2 ##################
ar15.power_spectrum()

plt.axis([0.2, 1.2, -40, 40])
plt.show()

"""Ex_B

Linear Model
"""

import numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def Linear_Regression_Model(x_pts, y_pts, order=1):
    d = {}
    d['x' + str(0)] = np.ones([1, len(x_pts)])[0]
    for i in np.arange(1, order + 1):
        d['x' + str(i)] = x_pts ** (i)

    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    X = np.column_stack(d.values())

    theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), y_pts)

    plt.figure()
    plt.scatter(x_pts, y_pts, s=30, c='b')
    line = theta[0]  # y-intercept
    label_holder = []
    label_holder.append('%.*f' % (2, theta[0]))
    for i in np.arange(1, len(theta)):
        line += theta[i] * x_pts ** i
        label_holder.append(' + ' + '%.*f' % (2, theta[i]) + r'$x^' + str(i) + '$')

    plt.plot(x_pts, line, label=''.join(label_holder))
    plt.title('Polynomial Fit: Order ' + str(len(theta) - 1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()


x_pts = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
y_pts = np.asarray(
    [3.0291, 3.2596, 1.5956, 0.9707, 3.1848, 2.4425, 2.7851, 4.3636, 4.8746, 5.6727, 7.8303, 8.9823, 9.9798, 10.9521,
     13.1084, 17.2846, 17.0282, 21.3780, 22.4312, 28.5930, 31.8417])


plt.figure()
plt.scatter(x_pts, y_pts)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

Linear_Regression_Model(x_pts, y_pts, order=2)
Linear_Regression_Model(x_pts, y_pts, order=3)
Linear_Regression_Model(x_pts, y_pts, order=4)

"""Non_Linear_Model"""

import numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style


def Non_Linear_Regression_Model(x_pts, y_pts, order=1):
    plt.figure()
    plt.scatter(x_pts, y_pts, s=30, c='b')

    d = {}
    d['x' + str(0)] = np.ones([1, len(x_pts)])[0]
    for i in np.arange(1, order + 1):
        d['x' + str(i)] = x_pts ** i
    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    X = np.column_stack(d.values())

    theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), np.log(y_pts))

    print ("theta is: ", theta)

    final_line = np.exp(theta[0]) * np.exp(theta[1] * x_pts)

    plt.plot(x_pts, final_line)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.yscale("log")
    plt.legend(loc='best')
    plt.show()


x_pts = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
y_pts = np.asarray(
    [3.0291, 3.2596, 1.5956, 0.9707, 3.1848, 2.4425, 2.7851, 4.3636, 4.8746, 5.6727, 7.8303, 8.9823, 9.9798, 10.9521,
     13.1084, 17.2846, 17.0282, 21.3780, 22.4312, 28.5930, 31.8417])


plt.figure()
plt.scatter(x_pts, y_pts)
plt.xlabel('x')
plt.ylabel('y')
plt.yscale("log")
plt.show()

Non_Linear_Regression_Model(x_pts, y_pts, order=1)
Non_Linear_Regression_Model(x_pts, y_pts, order=2)