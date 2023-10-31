import matplotlib
from matplotlib import pyplot as plt
from torch import Tensor

matplotlib.use("Agg")


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)

    if titles is None:
        titles = [None for i in range(len(data))]

    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]

        if isinstance(mel, Tensor):
            mel = mel.detach().cpu().numpy()

        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig
