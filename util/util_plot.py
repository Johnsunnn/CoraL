import numpy as np

import torch
import pandas as pd

import umap
import pickle

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def draw_umap_plot(repres_list, label_list, title, name):
    # plt.rcParams['savefig.dpi'] = 800  # 图片像素
    # plt.rcParams['figure.dpi'] = 800  # 分辨率
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    cmap = ListedColormap(['#00beca', '#f87671'])
    repres = np.array(repres_list.cpu())
    label = np.array(label_list.cpu())
    scaled_data = StandardScaler().fit_transform(repres)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_data)
    colors = np.array(["#00beca", "#f87671"])
    # print(embedding)
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=label, cmap=cmap, s=5
    )
    # l1, = plt.plot([], [], 'o', color='#00beca', label='positive')
    # l2, = plt.plot([], [], 'o', color='#f87671', label='negative')
    # global flag
    # if flag:
    #     plt.legend(bbox_to_anchor=(-0.15, 1.1), loc=1, borderaxespad=0)
    #     flag=False
    # plt.legend(loc='best')

    # fig, ax = plt.subplots()
    # # title = "The number of different protein residues(Train)"
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontdict={'weight': 'normal', 'size': 16})
    plt.xlabel(name, fontdict={'weight': 'normal', 'size': 16})

    cbar = plt.colorbar(sc, ticks=[0, 1])
    cbar.ax.set_yticklabels(['pos', 'neg'],fontdict={'weight': 'normal', 'size': 14})  # horizontal colorbar
    # plt.savefig('../umap/'+name+'.pdf')
    plt.show()