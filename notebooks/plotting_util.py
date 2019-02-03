import seaborn as sns
import numpy as np
import os
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import HTML
import matplotlib.pyplot as plt

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
    
    
    
    
    
def scatter_animation_2D(X):
    fig, ax = plt.subplots()
    num_timesteps = X.shape[0]

    ax.set_xlim((np.min(X[:,0]), np.max(X[:,0])))
    ax.set_ylim((np.min(X[:,1]), np.max(X[:,1])))
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')

    def update_plot(i, data, scatter):
        past_points = min(i,5)
        scatter.set_offsets(X[i-past_points:i,:])
        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]

        scatter.set_color(c)

        return scatter,

    scatter = plt.scatter([], [], s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps,
                                  fargs=(X, scatter))
    return ani

def scatter_animation_3D(Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3D')
    num_timesteps = Y.shape[0]

    ax.set_xlim((np.min(Y[:,0]), np.max(Y[:,0])))
    ax.set_ylim((np.min(Y[:,1]), np.max(Y[:,1])))
    ax.set_zlim((np.min(Y[:,2]), np.max(Y[:,2])))
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')

    def update_plot(i, data, scatter):
        past_points = min(i,5)
        scatter._offsets3d = (Y[i-past_points:i,0], Y[i-past_points:i,1],Y[i-past_points:i,2])
        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]
        scatter._facecolor3d = c
        scatter._edgecolor3d = c
        scatter.set_color(c)

        return scatter,

    scatter = ax.scatter([], [], [],s=100,depthshade=False)

    ani = animation.FuncAnimation(fig, update_plot, frames=50,
                                  fargs=(Y, scatter))
    return ani

def scatter_animation_2D_and_3D(X, Y):
    fig = plt.figure(figsize=(10,5))
    X_ax = fig.add_subplot(121,)
    Y_ax = fig.add_subplot(122, projection='3d')

    num_timesteps = Y.shape[0]

    X_ax.set_xlim((np.min(X[:,0]), np.max(X[:,0])))
    X_ax.set_ylim((np.min(X[:,1]), np.max(X[:,1])))
    X_ax.set_xlabel('Latent dim 1')
    X_ax.set_ylabel('Latent dim 2')


    Y_ax.set_xlim((np.min(Y[:,0]), np.max(Y[:,0])))
    Y_ax.set_ylim((np.min(Y[:,1]), np.max(Y[:,1])))
    Y_ax.set_zlim((np.min(Y[:,2]), np.max(Y[:,2])))
    Y_ax.set_xlabel('Obs dim 1')
    Y_ax.set_ylabel('Obs dim 2')
    Y_ax.set_zlabel('Obs dim 3')

    def update_plot(i, X, Y, X_scatter, Y_scatter):
        past_points = min(i,5)

        X_scatter.set_offsets(X[i-past_points:i,:])    
        Y_scatter._offsets3d = (Y[i-past_points:i,0], Y[i-past_points:i,1],Y[i-past_points:i,2])

        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]
        X_scatter.set_color(c)
        Y_scatter._facecolor3d = c
        Y_scatter._edgecolor3d = c

        return X_scatter, Y_scatter,

    X_scatter = X_ax.scatter([], [], s=100)
    Y_scatter = Y_ax.scatter([], [], [],s=100,depthshade=False)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps,
                                  fargs=(X, Y, X_scatter, Y_scatter))

    return ani

def plot_vector_field(*args):
    num_plots = len(args)
    fig, ax = plt.subplots(1,num_plots,figsize=(4*num_plots, 4))
    ax = np.atleast_1d(ax)
    xlims = [-2, 2]
    ylims = [-2, 2]
    X1, X2 = np.meshgrid(np.linspace(xlims[0], xlims[1], 10), np.linspace(ylims[0], ylims[1], 10))
    points = np.stack((X1, X2))
    for i, A in enumerate(args):
        AX = np.einsum('ij,jkl->ikl', A, points)
        Q = ax[i].quiver(X1, X2, AX[0] - X1, AX[1] - X2, units='width')      
        ax[i].set_xlim(xlims)
        ax[i].set_ylim(ylims)
    remove_frame(ax)
