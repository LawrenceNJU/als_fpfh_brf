""" Feature Visulization
Usage:
    feature_visualization <input_fname> <output_path> [-f=<feature_type>]
                                                      [--dpi=<figure_dpi>]
    feature_visualization -h | --help
    feature_visualization --version
Options:
    -h --help   
                        The <input_fname> argument must be the path to a point cloud file with pfh or fpfh feature.
                        The <output_path> argument must be a path to a diretory for storing the feature figure.
    -f=<feature_type> 
                        feature type contained in point cloud: pfh or fpfh; default: fpfh
    --dpi=<figure_dpi>
                        If given, the plt use tthis dpi  to output the figure

"""

import numpy as np
import pandas as ps
import matplotlib.pylab as plt
import matplotlib as mpl
from cycler import cycler
from docopt import docopt
import os, sys, shutil

if __name__ == "__main__":
    # arguments = docopt(__doc__, version='0.1.1rc')
    # print(arguments)
    opts = docopt(__doc__)

    feature_path = opts["<input_fname>"]
    output_path = opts["<output_path>"]

    feature_type = opts['-f'] if opts['-f'] else 'fpfh'
    dpi = opts['--dpi'] if opts['--dpi'] else 600
    # print(feature_path, output_path, feature_type, dpi)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)
    cloud = np.loadtxt(feature_path, skiprows=11)
    if feature_type == 'pfh':
        train_feature = cloud[:, 0:125]
        train_class = cloud[:, 125]
    elif feature_type == 'fpfh':
        train_feature = cloud[:, 0:33]
        train_class = cloud[:, 33]
    else:
        print("this feature cannot be visuliazed using the current code")
        exit(-1)
   
    class_name = ['Powerline', 'LowVegetation', 'ImperviousSurface', 'Car', 'Fence', 'Roof', 'Facade', 'Shrub', 'Tree']
    feature_class = np.hstack((train_feature, train_class.reshape([train_class.shape[0],1])))
    plt.style.use('default')
    mpl.rcParams['figure.figsize'] = 8.4, 9.6
    mpl.rcParams['axes.labelsize'] = 12
    # visualize feature
    for i in range(len(class_name)):
        a = feature_class[:, -1] == i
        feature = feature_class[a]
        feature_std = np.std(feature[:, :-1], axis =0)
        feature_mean = np.mean(feature[:, :-1], axis=0)
        sample_feature = feature[np.random.choice(feature.shape[0], 100, replace = False), : -1]

        plt.rc('lines', linewidth=0.3)
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.plot(np.transpose(sample_feature))
        ax0.set_ylim(0, 120)

        barWidth = 0.8
        plot_x = np.arange(len(feature_mean))
        ax1.set_ylim(0, 120)
        #ax1.bar(plot_x, fpfh_roof_mean, barWidth, color='cyan', yerr=fpfh_roof_std, ecolor = 'green', capsize=5)
        ax1.plot(plot_x, feature_mean, color = '#CEFC86', lw = 3 , label = 'fpfh')
        ax1.errorbar(plot_x, feature_mean, yerr = feature_std, ecolor = '#90AEFE', elinewidth = 2, capsize = 4, label = 'error bar')
        ax1.legend(loc = "best", fontsize =15, edgecolor = None)
        fig.savefig(output_path + '/' + class_name[i] + '.jpg', dip = dpi)
