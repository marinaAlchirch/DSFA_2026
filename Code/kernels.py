import numpy as np

class CustomGaussianKernel:

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def get_kernel_value(self, z):
        kernel_value = (1 / (np.sqrt(2 * np.pi) * self.sigma)) * np.exp(- (((z ** 2) / (2 * (self.sigma ** 2)))))
        return kernel_value


class CustomEpanechnikovKernel:

    def __init__(self):
        pass

    def get_kernel_value(self, z):
        kernel_value = max(0,1-(z**2))
        return kernel_value