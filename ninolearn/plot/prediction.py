import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(time, mean, std=None, std_level=1, facecolor='royalblue', line_color="navy"):
    if std is not None:
        p1std = mean + np.abs(std)
        m1std = mean - np.abs(std)
        p2std = mean + 2 * np.abs(std)
        m2std = mean - 2 * np.abs(std)

        plt.fill_between(time,m1std, p1std , facecolor=facecolor, alpha=0.7)
        plt.fill_between(time,m2std, p2std , facecolor=facecolor, alpha=0.3)

    plt.plot(time, mean, line_color)