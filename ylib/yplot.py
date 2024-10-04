import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import cv2
# from .scipy_misc import toimage

def plot_matrix(mat, path=None, xticks=None, yticks=None, xlim=None, ylim=None, figsize=(6,4), title=None, xlabel=None, ylabel=None, fontsize=20, cmap="YlGnBu"):
    plt.figure(figsize=figsize)
    # vis = vote_map.reshape(20, -1, 4).max(2)
    with sns.axes_style("white"):
        ax = sns.heatmap(mat, cmap=cmap)
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=fontsize)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=fontsize)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()




def plot_shaded(mid, high, low, path=None, xticks=None, yticks=None, xlim=None, ylim=None, figsize=(6,4), title=None, xlabel=None, ylabel=None, fontsize=20, color = '#5f87bc'):
    dim = len(mid)
    plt.figure(figsize=figsize)
    lw = 1
    plt.plot(mid, linewidth=lw, color=color)
    plt.plot(np.zeros(dim), linewidth=1/2., color=color)
    plt.fill_between(range(dim), low, high, linewidth=0.1, alpha=0.5, color=color)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

def plot_mean_std(mean, std, path=None, xticks=None, yticks=None, xlim=None, ylim=None, figsize=(6,4), title=None, xlabel=None, ylabel=None):
    plot_shaded(mean, mean+std, mean-std, path=path, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, figsize=figsize, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_lines(lines, path=None, legends=[], xticks=None, yticks=None, xlim=None, ylim=None, figsize=(6,4), linewidth=1, title=None, xlabel=None, ylabel=None, fontsize=20):
    # color = '#5f87bc'
    plt.figure(figsize=figsize)
    if len(legends) > 0:
        for line, legend in zip(lines, legends):
            plt.plot(line, label=legend, linewidth=linewidth)
        plt.legend()
    else:
        plt.plot(lines, linewidth=linewidth)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

def plot_distrib(dist1, dist2=None, path=None, figsize=(6,4), xticks=None, yticks=None, xlim=None, ylim=None, linewidth=1, title=None, xlabel=None, ylabel=None, fontsize=20, fpr_shade=True, color='#5f87bc'):
    plt.figure(figsize=figsize)
    g1 = sns.distplot(pd.Series(data=dist1, name=''), color=color, hist=False, kde_kws={"shade": True, 'linewidth': linewidth})
    if xlim is not None:
        g1.set(xlim=xlim)
    # g1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlim=xlim, ylim=ylim)
    # g1.set(xlabel=None, ylabel=None)
    if dist2 is not None:
        g2 = sns.distplot(pd.Series(data=dist2, name=''), color='#444444', hist=False, kde_kws={"shade": True, 'linewidth': linewidth},)
        # g2.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlim=xlim, ylim=ylim)
        # g2.set(xlabel=None, ylabel=None)
        if fpr_shade:
            arr = g2.get_children()[1].get_paths()[0].vertices
            x, y = arr[:, 0], arr[:, 1]
            x, y = x[y > 0], y[y > 0]
            x, y = x[x.argsort()], y[x.argsort()]
            mask = x > np.percentile(dist1, 5)
            g2.fill_between(x[mask], y1=y[mask], y2=0, alpha=0.3, facecolor='#444444', hatch='////')
    # sns.despine(bottom=True, left=True)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


# def colormap(img, mode=cv2.COLORMAP_JET):
#     img = toimage(img)
#     colormask = cv2.applyColorMap(np.array(img), mode)[:,:,::-1]
#     return colormask