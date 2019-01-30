import seaborn as sns
import numpy as np
import os

def xkcd_colors():
    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple",
                   "orange",
                   "clay",
                   "pink",
                   "greyish",
                   "mint",
                   "light cyan",
                   "steel blue",
                   "forest green",
                   "pastel purple",
                   "salmon",
                   "dark brown"]

    colors = sns.xkcd_palette(color_names)
    return colors

def remove_frame(ax_array,all_off=False):
    for ax in np.ravel(ax_array):
        if not all_off:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
        if all_off:
            ax.set_axis_off()


def savefig(fig, title, save_path='../figures/'):
    ''' Formats title and automatically saves in directory
    '''
    fig.savefig(os.path.join(save_path, '{}_{}.pdf'.format(datetime.now().date(), title)), bbox_inches='tight', transparent=True)