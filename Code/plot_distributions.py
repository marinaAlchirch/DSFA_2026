import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_distributions(kernel, kernel_distr_dict, h_list, emp_density_normalized, labels, directory, fig_name):
    x_axis = np.arange(len(labels))
    plt.figure(figsize=(10, 6))


    # Define color palette
    palette = sns.color_palette("husl", len(h_list))  # or try "tab10", "coolwarm", etc.

    # Plot original distribution
    plt.bar(x_axis, emp_density_normalized, color='blue', alpha=0.6, label='Original Normalized Count')


    counter = 0

    for h, smoothed_weights in kernel_distr_dict.items():
        plt.plot(x_axis, smoothed_weights,
                 label=f'KDE with {kernel} kernel and Bandwidth : {h}',
                 marker='o',
                 linewidth=2.5,
                 alpha=0.8,
                 color=palette[counter],
                 markeredgecolor='white',
                 markeredgewidth=0.5)
        counter += 1

    # Styling
    sns.set(font_scale=2)
    plt.xlabel('Label', fontsize=16)
    plt.ylabel('Normalized Count / Normalized KDE Weight', fontsize=16)
    plt.title("Empirical and KDE Curves for Different Bandwidths", fontsize=16)
    plt.xticks(ticks=x_axis, labels=labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=13)

    plt.tight_layout()

    # Save & Show
    plt.savefig(directory+fig_name+'.png')