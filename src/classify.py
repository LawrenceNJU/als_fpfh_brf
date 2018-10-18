""" Classify using multiscale pfh1 feature
Usage:
    classify <output_path>
    feature_visualization -h | --help
    feature_visualization --version
Options:
    -h --help   
                        The <output_path> argument must be a path to a diretory for storing the confusion matrix and so on.

"""

import pandas as ps
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import gc
import os, shutil

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize = True'
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    #print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, rotation_mode='anchor', ha='right', fontsize = 'large')
    plt.yticks(tick_marks, classes, fontsize = 'large')
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black", fontsize = 13)
    plt.ylabel('True label', fontsize = 15)
    plt.xlabel('Predicted label', fontsize = 15)


if __name__ == "__main__":
    class_name =  ['Powerline', 'Low Vegtation', 'Impervious Surface', 'Car', 'Fence', 'Roof', 'Facade', 'Shrub', 'Tree']
    
    """
    # train data with multiscale pfh feature
    train_cloud_1 = np.loadtxt('Vaihingen3D_Training_asZ_pfh1.txt')
    train_feature_1 = train_cloud_1[: , 0:125]
    train_class = train_cloud_1[ : , 125]
    normal = train_cloud_1[:, 130:134]
    vertical = np.array([0, 0, 1])
    angle = np.dot(normal, vertical)
    angle = angle.reshape([angle.shape[0], 1])

    train_cloud_2 = np.loadtxt('Vaihingen3D_Training_asZ_pfh2.txt')
    train_feature_2 = train_cloud_2[: , 0:125]
    del train_cloud_2
    gc.collect()

    train_cloud_3 = np.loadtxt('Vaihingen3D_Training_asZ_pfh3.txt')
    train_feature_3 = train_cloud_3[: , 0:125]
    del train_cloud_3
    gc.collect()

    train_feature_all = np.hstack((train_feature_1, train_feature_2))
    train_feature_all = np.hstack((train_feature_all, train_feature_3))
    train_feature_all = np.hstack((train_feature_all, train_cloud_1[ : ,134]))
    train_feature_all = np.hstack((train_feature_all, angle))
    del angle, train_cloud_1, train_feature_1, train_feature_2, train_feature_3
    gc.collect()

    # test data with multiscale pfh feature
    test_cloud_1 = np.loadtxt('Vaihingen3D_Testing_asZ_pfh1.txt')
    test_feature_1 = test_cloud_1[: , 0:125]
    test_class = test_cloud_1[ : , 125]

    test_cloud_2 = np.loadtxt('Vaihingen3D_Testing_asZ_pfh2.txt')
    test_feature_2 = test_cloud_2[: , 0:125]
    del test_cloud_2
    gc.collect()

    test_cloud_3 = np.loadtxt('Vaihingen3D_Testing_asZ_pfh3.txt')
    test_feature_3 = test_cloud_3[: , 0:125]
    del test_cloud_3
    gc.collect()

    test_feature_all = np.hstack((test_feature_1, test_feature_2))
    test_feature_all = np.hstack((test_feature_all,test_feature_3))
    test_feature_all = np.hstack((test_feature_all, test_cloud_1[ : , np.array([126, 129, 130, 131, 134])]))
    del test_cloud_1, test_feature_1, test_feature_2, test_feature_3
    gc.collect()
    """ 
    """
    train_cloud_3 = np.loadtxt('Vaihingen3D_Training_asZ_pfh3.txt')
    train_feature_3 = train_cloud_3[: , 0:125]
    train_class = train_cloud_3[ : , 125]
    normal = train_cloud_3[:, 129:132]
    vertical = np.array([0, 0, 1])
    angle = np.dot(normal, vertical)
    angle = angle.reshape([angle.shape[0], 1])
    normalized_height = train_cloud_3[:, 134]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])

    train_feature_all = np.hstack((train_feature_3, train_cloud_3[ : , np.array([126, 134])]))
    train_feature_all = np.hstack((train_feature_all, angle))
    del normal, angle, train_feature_3, normalized_height
    gc.collect()

    test_cloud_3 = np.loadtxt('Vaihingen3D_Testing_asZ_pfh3.txt')
    test_feature_3 = test_cloud_3[: , 0:125]
    test_class = test_cloud_3[ : , 125]
    normal = test_cloud_3[:, 129:132]
    angle = np.dot(normal, vertical)
    angle = angle.reshape([angle.shape[0], 1])
    normalized_height = test_cloud_3[:, 134]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])

    #test_feature_all = np.hstack((test_feature_3, normalized_height))
    test_feature_all = np.hstack((test_feature_3, test_cloud_3[ : , np.array([126, 134])]))
    test_feature_all = np.hstack((test_feature_all, angle))
    del normal, angle, test_cloud_3, test_feature_3, normalized_height
    gc.collect()
    """
    """
    # train data with multiscale fpfh feature
    train_cloud_1 = np.loadtxt('Vaihingen3D_Training_asZ_fpfh1.txt')
    train_feature_1 = train_cloud_1[: , 0:33]
    train_class = train_cloud_1[ : , 33]
    normal = train_cloud_1[:, 37:40]
    vertical = np.array([0, 0, 1])
    angle = np.dot(normal, vertical)
    angle = angle.reshape([angle.shape[0], 1])
    normalized_height = train_cloud_1[:, 42]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])

    train_cloud_2 = np.loadtxt('Vaihingen3D_Training_asZ_fpfh2.txt')
    train_feature_2 = train_cloud_2[: , 0:33]
    del train_cloud_2
    gc.collect()

    train_cloud_3 = np.loadtxt('Vaihingen3D_Training_asZ_fpfh3.txt')
    train_feature_3 = train_cloud_3[: , 0:33]
    del train_cloud_3
    gc.collect()

    train_feature_all = np.hstack((train_feature_1, train_feature_2))
    train_feature_all = np.hstack((train_feature_all, train_feature_3))
    train_feature_all = np.hstack((train_feature_all, normalized_height))
    train_feature_all = np.hstack((train_feature_all, angle))
    del angle, normalized_height, train_cloud_1, train_feature_1, train_feature_2, train_feature_3
    gc.collect()

    # test data with multiscale fpfh feature
    test_cloud_1 = np.loadtxt('Vaihingen3D_Testing_asZ_fpfh1.txt')
    test_feature_1 = test_cloud_1[: , 0:33]
    test_class = test_cloud_1[ : , 33]
    normal = test_cloud_1[:, 37:40]
    vertical = np.array([0, 0, 1])
    angle = np.dot(normal, vertical)
    angle = angle.reshape([angle.shape[0], 1])
    normalized_height = test_cloud_1[:, 42]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])
    
    test_cloud_2 = np.loadtxt('Vaihingen3D_Testing_asZ_fpfh2.txt')
    test_feature_2 = test_cloud_2[: , 0:33]
    del test_cloud_2
    gc.collect()

    test_cloud_3 = np.loadtxt('Vaihingen3D_Testing_asZ_fpfh3.txt')
    test_feature_3 = test_cloud_3[: , 0:33]
    del test_cloud_3
    gc.collect()

    test_feature_all = np.hstack((test_feature_1, test_feature_2))
    test_feature_all = np.hstack((test_feature_all, test_feature_3))
    test_feature_all = np.hstack((test_feature_all, angle))
    test_feature_all = np.hstack((test_feature_all, normalized_height))
    del angle, normalized_height, test_cloud_1, test_feature_1, test_feature_2, test_feature_3
    gc.collect()
    """
    
    # train data with fpfh_3 feature
    train_cloud_1 = np.loadtxt('Vaihingen3D_Training_asZ_fpfh2.txt')
    train_feature_1 = train_cloud_1[: , 0:33]
    train_class = train_cloud_1[ : , 33]
    normal = train_cloud_1[:, 37:40]
    # vertical = np.array([0, 0, 1])
    # angle = np.dot(normal, vertical)
    # angle = angle.reshape([angle.shape[0], 1])
    normalized_height = train_cloud_1[:, 42]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])

    # train_feature_all = np.hstack((train_feature_1, normalized_height))
    # train_feature_all = np.hstack((train_feature_all, normal))
    # del angle, normalized_height, train_cloud_1, train_feature_1
    # gc.collect()
    train_feature_all = train_feature_1


    # test data with multiscale fpfh feature
    test_cloud_1 = np.loadtxt('Vaihingen3D_Testing_asZ_fpfh2.txt')
    test_feature_1 = test_cloud_1[: , 0:33]
    test_class = test_cloud_1[ : , 33]
    normal = test_cloud_1[:, 37:40]
    # vertical = np.array([0, 0, 1])
    # angle = np.dot(normal, vertical)
    # angle = angle.reshape([angle.shape[0], 1])
    normalized_height = test_cloud_1[:, 42]
    normalized_height = normalized_height.reshape([normalized_height.shape[0], 1])
    
    test_feature_all = test_feature_1
    # test_feature_all = np.hstack((test_feature_1, normalized_height))
    # test_feature_all = np.hstack((test_feature_all, normal))
    # del angle, normalized_height, test_cloud_1, test_feature_1
    # gc.collect()

    # train random forest using feature and predict test data label
    clf = RandomForestClassifier (n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(train_feature_all, train_class)
    pre_test_class = clf.predict(test_feature_all)

    output_path = './onlywith_fpfh_classify'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)

    joblib.dump(clf, output_path + '/' + 'classify.pkl') 
    
    with open(output_path + '/' + 'classifaction_report.txt', 'wt') as f:
        print(classification_report(test_class, pre_test_class, target_names=class_name), file=f)

    confusion_mat = confusion_matrix(test_class, pre_test_class)
    fig = plt.figure()
    plot_confusion_matrix(confusion_mat, classes = class_name, title = 'Confusion matrix, without normalization')
    fig.savefig(output_path + '/' + 'confusion_matrix.jpg', dip=600)
    with open(output_path + '/' + 'confusion_mat.txt', 'wt') as f:
        print(confusion_mat, file=f)
