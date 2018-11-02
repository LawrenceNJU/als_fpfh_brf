import numpy as np
import collections

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble
from sklearn.neighbors import KDTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

class_name =  ['Powerline', 'Low Vegetation', 'Impervious Surface', 'Car', 'Fence', 'Roof', 'Facade', 'Shrub', 'Tree']

def feature_extraction(cloud):
    
    #calculathe the fpfh data
    fpfh = cloud[:, 1:34]
    
    # calculate the normal vector
    normal = cloud[:, 38:41]
    
    # calculate relative_height
    ground_xyz = cloud[cloud[:,0].astype(int)==1,-3:]
    kdt = KDTree(ground_xyz[:, 0:2], metric = 'euclidean')
    ind = kdt.query(cloud[:, -3:-1], k=1, return_distance = False)
    relative_height = cloud[:, -1] - ground_xyz[ind.flatten(), -1]
    relative_height = relative_height.reshape([cloud.shape[0], 1])
    
    # compose feature 
    feature = np.hstack((fpfh, normal))
    feature = np.hstack((feature, relative_height))
    return feature

# RBF classify the data
def resample_data(train_feature, train_class, count_sampleset):
    
    multiplier = {0: 1.0, 1: 0.1, 2: 0.1, 3: 1.0, 4: 1.0, 5: 0.1, 6: 1.0, 7:0.5, 8: 0.1}
    target_stats = collections.Counter(train_class)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    
    ee = EasyEnsemble(ratio=target_stats ,n_subsets=count_sampleset)
    return ee.fit_sample(train_feature, train_class)
    
def train_brf(X_resampled, y_resampled, count_learnbase):
    # generalize dicision tree
    random_state =42 # in order to every time have the same random discision tree for same data set 
    random_state = check_random_state(random_state)
    random_state = random_state.rand(count_learnbase)
    random_state = random_state * 1000000000
    random_state = random_state.astype('int')
    
    clf_estimator = []

    for i in range(count_learnbase):
        tmp_clf = DecisionTreeClassifier(max_features='auto', random_state=random_state[i])
        tmp_clf.fit(X_resampled[i], y_resampled[i])
        clf_estimator.append(tmp_clf)
    
    return clf_estimator
    
def predict_brf(clf_estimator, X):
    first_learnbase = True
    for clf in clf_estimator:
        if first_learnbase:
            predict_X = clf.predict(X)
            first_learnbase = False
        else:
            tmp_predict = clf.predict(X)
            predict_X = np.vstack((predict_X, tmp_predict))
    predict = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predict_X.astype('int'))
    return predict

if __name__ == "__main__":
    for radius in range(2, 16, 1):
        train_data = np.loadtxt('./Vaihingen/fpfh_ground/Vaihingen3D_Traininig_fpfh_{}_ground.txt'.format(radius), skiprows=11)
        train_data_feature = feature_extraction(train_data)
        train_data_class = train_data[:, 34]
        
        test_data = np.loadtxt('./Vaihingen/fpfh_ground/Vaihingen3D_EVAL_WITH_REF_fpfh_{}_ground.txt'.format(radius), skiprows=11)
        test_data_feature = feature_extraction(test_data)
        test_data_class = test_data[:, 34]
        
        count_learnbase = 50
        X_resampled, y_resampled = resample_data(train_data_feature, train_data_class, count_learnbase)
        clf_estimator = train_brf(X_resampled, y_resampled, count_learnbase)
        test_predict = predict_brf(clf_estimator, test_data_feature)
        precision_recall_fscore = precision_recall_fscore_support(test_data_class, test_predict)
        precision_recall_fscore_average = precision_recall_fscore_support(test_data_class, test_predict, average='weighted')
        for i in range(3):
            if i == 0:
                measure = np.append(precision_recall_fscore[i], precision_recall_fscore_average[i])
            else:
                tmp = np.append(precision_recall_fscore[i], precision_recall_fscore_average[i])
                measure = np.vstack((measure, tmp))
        if radius == 2:
            final_measure = measure
        else:
            final_measure = np.dstack(final_measure, measure)
    np.save('./classify_output/final_measure.npy', final_measure)