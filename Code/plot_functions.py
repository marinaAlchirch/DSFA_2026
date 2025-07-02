import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_curve_across_all_models(score_list_model1, score_list_model2, score_list_model3,
                                                     score_list_model4, score_name="MSE"):

    sns.set_theme()

    plt.figure(figsize=(12, 8))

    # Plot without jitter
    plt.plot(score_list_model1, label='HT', color='blue', linewidth=1.5, linestyle='-', alpha=0.8)
    plt.plot(score_list_model2, label='HT + HS', color='green', linewidth=1.5, linestyle='--', alpha=0.8)
    plt.plot(score_list_model3, label='HT + LDS', color='orange', linewidth=1.5, linestyle='-.', alpha=0.8)
    plt.plot(score_list_model4, label='HT + LDS + HS', color='red', linewidth=1.5, linestyle=':', alpha=0.8)

    plt.title(f"{score_name} curve across all 4 Models", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Instances Seen", fontsize=18)
    plt.ylabel(score_name, fontsize=18)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()

    # Save the plot
    save_path = '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/figures/california/with_bins/best_hyperparameters_new/' + str(score_name) + '.png'
    plt.savefig(save_path)

    plt.show()


