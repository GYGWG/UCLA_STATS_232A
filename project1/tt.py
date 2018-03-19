import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import scipy.optimize
import math
from scipy.special import gamma

#
# mean, var, beta = 1.0, 2.0, 0.9
# data = scipy.stats.gennorm.rvs(beta=beta, loc=mean, scale=var, size=1000)
# data2 = scipy.stats.gennorm.rvs(beta=beta, loc=mean * 2, scale=var * 2, size=1000)
# plt.subplot(121)
# hist, edges, patach = plt.hist(data, 1000)
# plt.subplot(122)
# plt.hist(data2, 1000)
# plt.show()
# param1 = scipy.stats.gennorm.fit(data)
# param2 = scipy.stats.beta.fit(data)
# param3 = scipy.stats.norm.fit(data)
#
# param = [0.21860928491444342, -1.9569067977309827e-29, 0.0003665023486286573]   # Beta, loc, scale
# def GGD(x, alpha, beta):
#     return beta/(2 * alpha * gamma(1/beta)) * np.exp(-(np.abs(x)/alpha) ** beta)
#     return np.exp(-(np.abs(x)/alpha) ** beta)
#
# popt, pcov = scipy.optimize.curve_fit(GGD, edges[:-1], hist)
#
# x = np.linspace(scipy.stats.gennorm.ppf(0.001, *param), scipy.stats.gennorm.ppf(0.999, *param), 500)
# x = np.linspace(-20,20,1000)
# y = GGD(x, param[2], param[0])
#
# fig, ax = plt.subplots(1,1)
# ax.plot(x, scipy.stats.gennorm.pdf(x, *param), 'r-', label='gennorm pdf')
# ax.plot(x, scipy.stats.gennorm.pdf(x, param[0]), 'g-', label='gennorm pdf2')
# ax.plot(x, y, 'k-', label='GGD')
# # ax.hist(data, log=True, normed=True, histtype='stepfilled')
# plt.legend()
# plt.show()

a = np.arange(21).reshape([7,-1])
