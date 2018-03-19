# Reference:
#   1. https://stackoverflow.com/questions/32694007/opencv-python-how-to-change-image-pixels-values-using-a-formula
#   2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html
#   3. https://stackoverflow.com/questions/44003552/matplotlib-histogram-from-numpy-histogram-output
#   4. http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html#fourier-transform
#   5. https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite


import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy.special import gamma
from math import sqrt


#
# Problem1
#

class Problem1:
    def __init__(self, imgFile):
        """
        Initialization
        :param imgFile: ndarray or str
        """
        # Load image in greyscale and re-scale the intensity to [0, 31]
        if type(imgFile).__module__ == np.__name__:
            self.img = imgFile.astype(np.uint8)
            imgFile = "q6"
        else:
            self.img = cv2.imread(imgFile, 0)
            self.rows, self.cols = self.img.shape
            self.img = self.img // 8

        self.showImg(self.img, "img")
        cv2.imwrite("{}_preprocessing1.png".format(imgFile), self.img)
        self.adjDiff = self.gradientFilter(self.img.astype(np.int16))
        self.showImg(self.adjDiff.astype(np.uint8), "adjDiff")
        cv2.imwrite("{}_preprocessing2.png".format(imgFile), self.adjDiff.astype(np.uint8))

    def showImg(self, img, imgName):
        """
        Present the image
        :param img: ndarray
        :param imgName: str
        :return: None
        """
        cv2.namedWindow(imgName, cv2.WINDOW_NORMAL)
        cv2.imshow(imgName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def gradientFilter(self, img, type = "adjDiff", param = 1):
        """
        Use different type of gradient filter to convolve the image
        :param img: ndarray
        :param type: str
        :param param: tuple.
        :return: int16 ndarray
        """
        if type == "adjDiff":
            newImg = cv2.filter2D(img, -1, np.array([-1.0, 1.0]))
            newImg[0,:] = newImg[1,:]
            return newImg
        elif type == "laplacian":
            return cv2.Laplacian(img, -1, ksize=param)
        elif type == "sobelx":
            return cv2.Sobel(img, -1, 1, 0, ksize=param)
        elif type == "sobely":
            return cv2.Sobel(img, -1, 0, 1, ksize=param)
        else:
            print "Wrong type name."
            return None

    def GGD(self, x, alpha, beta):
        """
        Generalized Gaussian Distribution
        :param x: ndarray
        :param alpha: float
        :param beta: float
        :return: ndarray
        """
        return beta/(2 * alpha * gamma(1/beta)) * np.exp(-(np.abs(x) / alpha) ** beta)

    def GD(self, x, mean, stdDev):
        return 1/sqrt(2 * np.pi) / stdDev * np.exp(-((x - mean)/stdDev) ** 2)

    def q1(self, label = "H(z)"):
        print "STEP 1"
        self.diffZ = self.adjDiff.astype(np.int8)
        plt.subplots(1,1)
        self.histr, self.edges, patch = plt.hist(self.diffZ.ravel(), 63, [-31, 31], normed=True, label="histogram {}".format(label))
        plt.legend()
        plt.show()

        self.log10Histr = np.log10(self.histr[self.histr != 0])
        self.logEdges = self.edges[:-1][self.histr != 0]
        plt.subplots(1, 1)
        # ax2.bar(edges[:-1], np.log10(histr), width=np.diff(edges), ec = None, align = 'edge')
        plt.plot(self.logEdges, self.log10Histr, 'b--', label="log histogram log{}".format(label))
        plt.xlim([-31,32])
        plt.legend()
        plt.show()

    def q2(self):
        self.mean, self.var, self.kur = np.mean(self.diffZ.ravel()), np.var(self.diffZ.ravel()), scipy.stats.kurtosis(self.diffZ.ravel())
        print "STEP 2: mean = {}, var = {}, kur = {}".format(self.mean, self.var, self.kur)

    def q3(self):
        self.param, pcov = curve_fit(self.GGD, self.edges[:-1], self.histr)
        print "STEP 3: sigma = {}, beta = {}".format(*self.param)
        # param2 = scipy.stats.gennorm.fit(diffZ.ravel(), loc=0.0)
        x = np.linspace(-32,32, 1000)
        y_GGD = self.GGD(x, *self.param)
        plt.hist(self.diffZ.ravel(), 63, [-31, 31], normed=True)
        plt.plot(x, y_GGD, 'k-', label='GGD')
        # plt.plot(x, scipy.stats.gennorm.pdf(x, *param2), 'r-', label='gennorm pdf')
        plt.xlim([-31,32])
        plt.legend()
        plt.show()

    def q4(self, label = "H(z)"):
        print "STEP 4"
        x = np.linspace(-32,32, 1000)
        y_GD = self.GD(x, self.mean, sqrt(self.var))
        plt.hist(self.diffZ.ravel(), 63, [-31, 31], normed=True, label="histogram {}".format(label))
        # plt.plot(x, scipy.stats.norm.pdf(x, 0, param1[0]), 'r-', label='Gaussian distribution')
        plt.plot(x, y_GD, 'r-', label='Gaussian distribution')
        plt.xlim([-31,32])
        plt.legend()
        plt.show()

        plt.plot(self.logEdges, self.log10Histr, 'k--', label="log histogram log{}".format(label))
        plt.plot(x[y_GD!=0], np.log10(y_GD[y_GD!=0]), 'r-', label='log Gaussian distribution')
        plt.xlim([-31, 32])
        plt.legend()
        plt.show()

    def q5(self, label = "H(z)"):
        print "STEP 5"
        rows, cols = self.img.shape
        times = ["first", "second", "third"]
        lineType = ['b--', 'g-.', 'r:']
        f, ax = plt.subplots(1, 2)
        ax[0].plot(self.edges[:-1], self.histr, 'k-', label="log histogram logH(z) of Step1")
        ax[1].plot(self.logEdges, self.log10Histr, 'k--', label="log histogram logH(z) of Step1")
        fHist, axHist = plt.subplots(1,1)
        DSdiffZ = self.img

        for ind in xrange(2): # Do the downsampling 2 times
            DSdiffZ = np.array([ [np.sum(DSdiffZ[2*i:2*(i+1), 2*j: 2*(j+1)]) / 4 for j in xrange(cols/2)] for i in xrange(rows/2) ])
            rows, cols = DSdiffZ.shape
            DSdiffZ = self.gradientFilter(DSdiffZ.astype(np.int16))

            histr, edges, patch = axHist.hist(DSdiffZ.ravel(), 63, [-31, 31], normed=True, label="histogram {} of {} downsampling".format(label, times[ind]))

            ax[0].plot(edges[:-1], histr, lineType[ind], label="histogram {} of {} downsampling".format(label, times[ind]))

            logHistr= np.log10(histr[histr != 0])
            logEdges = edges[:-1][histr != 0]
            ax[1].plot(logEdges, logHistr, lineType[ind], label="log histogram log{} of {} downsampling".format(label, times[ind]))

        ax[0].legend(), ax[1].legend()
        plt.xlim([-31, 32])
        plt.show()

    def q6(self):
        print "STEP 6"
        q6 = Problem1(np.random.randint(0,32, [self.rows, self.cols]))
        q6.q1(" of the uniform noise image")
        q6.q2()
        q6.q4(" of the uniform noise image")
        q6.q5(" of the uniform noise image")


class Problem2:
    def __init__(self, imgFile):
        self.img = cv2.imread(imgFile, 0)






pb1 = Problem1("natural_scene_1.jpg")
# pb1.q1()
# pb1.q2()
# pb1.q3()
# pb1.q4()
# pb1.q5()
pb1.q6()