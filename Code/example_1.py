from Code.kernels import CustomGaussianKernel, CustomEpanechnikovKernel
from Code.kde import kde
from Code.plot_distributions import plot_distributions

import numpy as np

def run_example():
    # TODO : Maybe we want to express the points in a dictionary of key : label, value : multiplicity of label, instead of a list
    z_list = [1, 1, 1, 3, 4, 4]
    h_list_gaussian = [0.1, 0.5, 0.7, 1, 5]
    h_list_epanechnikov = [1, 2, 3, 4, 5]
    labels = np.array([1, 2, 3, 4, 5])
    emp_density = np.array([3, 0, 1, 2, 0])
    emp_density_normalized = emp_density / np.sum(emp_density)
    x = np.arange(len(labels))

    # Run with Gaussian
    G_kde_h_w_dict = {}
    G_distr_h_w_dict = {}
    sigma = np.std(z_list)
    kernel = CustomGaussianKernel(sigma)
    for h in h_list_gaussian:
        kde_gaussian = []
        kde_distribution = []
        for z in labels:
            print(f"Label/Query {z}")
            print(str(kde(z_list, z, h, kernel)))
            kde_gaussian.append(kde(z_list, z, h, kernel))

        a = 1/np.sum(kde_gaussian)
        for i in range(len(kde_gaussian)):
            kde_distribution.append(kde_gaussian[i] * a)
        G_kde_h_w_dict[h] = kde_gaussian
        G_distr_h_w_dict[h] = kde_distribution
        print(f"KDE with bandwidth {h} : {G_kde_h_w_dict[h]}")
        print(f"KDE Distribution with bandwidth {h} : {G_distr_h_w_dict[h]}")

    fig_name = "example_direct_and_incr_lds_multiple_bandwidth"
    plot_distributions("Gaussian", G_distr_h_w_dict, h_list_gaussian, emp_density_normalized, labels, "/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/figures/Gaussian/", fig_name=fig_name)

    # Run with Epanechnikov
    E_kde_h_w_dict = {}
    E_distr_h_w_dict = {}
    kernel = CustomEpanechnikovKernel()
    for h in h_list_epanechnikov:
        kde_epanechnikov = []
        kde_distribution = []
        for z in labels:
            kde_epanechnikov.append(kde(z_list, z, h, kernel))

        a = 1 / np.sum(kde_epanechnikov)
        for i in range(len(kde_epanechnikov)):
            kde_distribution.append(kde_epanechnikov[i] * a)
        E_kde_h_w_dict[h] = kde_epanechnikov
        E_distr_h_w_dict[h] = kde_distribution

    plot_distributions("Epanechnikov", E_distr_h_w_dict, h_list_epanechnikov, emp_density_normalized, labels, "/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/figures/Epanechnikov/", fig_name=fig_name)


