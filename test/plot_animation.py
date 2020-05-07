import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def bar_plot(fig, gestures, prediction):
    plt.ion()
    ax = fig.add_subplot(111)

    ax.set_title('Real-time Predictions', fontsize=12)
    plt.xlabel('Human Activities', fontsize=10)
    plt.ylabel('', fontsize=10)
    x = range(0, len(gestures)-1, 1)
    plt.yticks(fontsize=10)
    plt.xticks(x, gestures, fontsize=10)

    colors = cm.rainbow(np.linspace(0, 1.0, len(gestures)))
    y_score = np.zeros((len(gestures),))
    x_score = np.linspace(0, len(gestures)-1, len(gestures))
    y_score[gestures.index(prediction)] = 1

    ax.bar(x_score, y_score, color=colors[gestures.index(prediction)])

    plt.pause(0.0001)
    plt.clf()

