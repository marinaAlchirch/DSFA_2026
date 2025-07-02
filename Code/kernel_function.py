import numpy as np

def custom_gaussian_kernel(z, sigma=1.0):
    kernel_value = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (((z ** 2) / (2 * (sigma ** 2)))))
    return kernel_value

def gaussian_kernel(z):
    # TODO : sigma = find_sigma()
    sigma = 1.0
    kernel_value = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (((z**2)/(2*(sigma**2)))))
    return kernel_value

def epanechnikov_kernel(z):
    return max(0,1-(z**2))