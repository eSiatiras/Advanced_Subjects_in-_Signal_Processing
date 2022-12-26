import matlab.engine
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal as sc
from scipy.io.wavfile import write
import scipy.io
import numpy as np

eng = matlab.engine.start_matlab()


class WienerFIRFilter(object):
    def __init__(self, u2, y, order):
        self.u2 = u2
        self.y = y
        self.lag = order

    def compute_coeffs(self):
        ac = np.asarray(eng.autocorr(matlab.double(list(self.u2)), self.lag))
        R = linalg.toeplitz(ac)
        r = np.asarray(eng.crosscorr(matlab.double(list(self.y)), matlab.double(list(self.u2)), self.lag))
        self.w_opt = np.dot(linalg.inv(R), r.T)
        self.w_opt = np.reshape(self.w_opt, -1)
        return self.w_opt

    def predict_noise(self, u2):
        self.u1 = sc.lfilter(self.w_opt, [1], u2)
        return self.u1

    def predict_signal(self, y):
        return np.subtract(y, self.u1)

    def plot_predicted_signal(self, s, y):
        plt.plot(s, 'b', label="Estimate s(n)")
        plt.title('The Predicted Signal s[n],Filter Order: ' + str(self.lag))
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.show()


mat = scipy.io.loadmat('C.mat', squeeze_me=False, chars_as_strings=False, mat_dtype=True,
                       struct_as_record=True)

u2 = np.reshape(mat['u'], -1)
y = np.reshape(mat['y'], -1)

plt.plot(y, 'r')
plt.title("Corrupted Signal y(n)")
plt.xlabel("Samples")
plt.ylabel("Values")
plt.show()

w = WienerFIRFilter(u2, y, 80)
w.compute_coeffs()
u1 = w.predict_noise(u2)
s = w.predict_signal(y)
w.plot_predicted_signal(s, y)

scipy.io.savemat('s.mat', {'s': s})
scipy.io.wavfile.write('result.wav', 8192, s)
