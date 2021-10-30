import os, pickle, datetime, operator
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from segmentation_library.Utilities.data_io import QualisysData, read_metadata,\
    get_segment_hierarchy, read_demojson, save_demo2json
from vmci_segmentation.preprocessing import normalize_differences

from segmentation_library.Utilities.visualization import animate_segmentation, plot_segmentation
from segmentation_library.segment_recognition.features import calc_features
from segmentation_library.MCI.run import vmci

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

SparseMatrix = {}

def calculate_features(demo, labeldict,
                       mean_len,
                       standardize=False,
                       normalize=True
                       ):
    
    """
    Method to calculate features for annotation. 

    Parameters
    ----------
    demo : list
        Demonstration to be annotated.  ''demo'' is a list of demos.
    
    labeldict : dict

    mean_len : int
        Length of all segments to be interpolated to. 
    
    standardize : bool

    normalize : bool

    Returns
    -------

    features : List of array-like, shape: (n_segments, n_features)
        Features of segments.
    labels : list

    """
    
    sensor_names = ['right_hand_top', 'right_hand_right', 'right_hand_left',
                    'right_elbow', 'right_shoulder',
                    'back_top', 'back_right', 'back_left']
    
    reference_sensors = ['back_top', 'back_right', 'back_left']
    position_sensors = ['right_hand_top', 'right_elbow', 'right_shoulder']
    angle_sensors = ['right_elbow', 'right_hand_top', 'right_shoulder', # elbow joint
                     'right_shoulder', 'right_elbow', 'back_top' # shoulder joint
                     ]
    orientation_sensors = ['right_hand_top', 'right_hand_right', 'right_hand_left']
 
    features, labels, _ = calc_features(demo, 
                                     sensor_names, 
                                     reference_sensors, 
                                     labeldict=labeldict,
                                     frequency = demo[0].frequency,   
                                     mean_len=mean_len,
                                     training=True,
                                     normalize=normalize,
                                     standardize=standardize,
                                     features=['position','velocity', 'joint_angle', 'joint_velocity',
                                               'orientation'],
                                     position_sensors = position_sensors,
                                     angle_sensors=angle_sensors,
                                     orientation_sensors = orientation_sensors, 
                                     )
    

    return features, labels

def run_feature_calculation(demo, suffix = r'_normalized'):
    
    """
    Calculate features based on the demonstration.

    Parameters
    ----------
    demo : list
        json file various sets of values.
    
    Returns
    -------
    features : list of array-like, shape: (n_segments, n_features)
        Features of segments.
    
    labels : list
        Labels for each of features.

    labeldict : dict
        Dictionary with labels for various movements.
    """

    movement_classes = ['middle2front','front2middle', 'middle2right', 'right2middle', 'middle2left', 'left2middle', 'middle2down' ,'down2middle'] 

    labeldict = {a: num for (a, num) in zip(movement_classes, range(len(movement_classes)))}
    
    normalize=False
    standardize=False
    
    if 'normalized' in suffix:
        normalize=True
    elif 'standardized' in suffix:
        standardize=True
   
    features, labels = calculate_features(demo, labeldict, mean_len = 40,
                       normalize=normalize, standardize=standardize)
    
    return features, labels, labeldict

def create_feat(features, labels):
    
    """
    Create training features provided training features and labels from the json file.

    Parameters
    ----------
    features : numpy array 
        train values.
    
    labels : list
        labels for train values.
    
    Returns
    ------- 
    features : list
        features for each training values.
    labels : list
        labels for each training values.
    """
    
    training_feat = None
    for index, (feat_1, feat_2) in enumerate(zip(features, features[1:])):
        train_feat = np.hstack((feat_1, feat_2))
        if training_feat is None:
            training_feat = train_feat
        else:    
            training_feat = np.vstack((training_feat, train_feat))
    train_labels = labels[2:]
    
    return training_feat[:-1], train_labels        
    

def train_Rf_model(X_train, y_train, cv_inner):
    
    """
    Train Random forest classifier model
    Parameters
    ----------
    X_train : numpy array
        train features
    y_train : list 
        train labels
    cv_inner : class of Kfold split
        validation with 5 folds
    Returns
    -------
    RF_grid : 
        rf grid with trained values
    """

    pipe = Pipeline([('RF', RandomForestClassifier())])
    params = {'RF__n_estimators':np.arange(5,100,5),'RF__max_depth':np.arange(4,100,4)}
    rfc_grid = GridSearchCV(pipe, param_grid=params, n_jobs = -1, cv=cv_inner)
    rfc_grid.fit(X_train, y_train)
    return rfc_grid

def train_KNN_model(X_train,y_train, cv_inner):
    
    """
    Train K Nearest Neighbor model
    
    Parameters
    ----------
    X_train : numpy array
        train features
    y_train : list
        train labels
    cv_inner : class of Kfold split
        validation with 5 folds
    Returns
    -------
    knn_grid : 
        knn grid with trained values
    
    """
    pipe = Pipeline([('KNN', KNeighborsClassifier())])
    params = {'KNN__n_neighbors':np.arange(1,5,2)}
    knn_grid = GridSearchCV(pipe, param_grid=params, n_jobs = -1, cv=cv_inner)   
    knn_grid.fit(X_train, y_train)
    return knn_grid

def train_XG_Boost_model(X_train, y_train, cv_inner):
    
    """
    Train XG Boost model
    
    Parameters
    ----------
    X_train : numpy array
        train features.
    y_train : list 
        train labels.
    cv_inner : class of Kfold split
        validation with 5 folds.

    Returns
    -------
    xgb_grid : 
        XG boost grid with trained values.
    """
    print(type(X_train))
    print(type(y_train))
    print(type(cv_inner))

    pipe = Pipeline([('XGB', XGBClassifier())])
    params = {'XGB__max_depth':np.arange(100,300,50), 'XGB__n_estimators':np.arange(100,200,20)}
    xgb_grid = GridSearchCV(pipe, param_grid=params, cv=cv_inner, n_jobs = -1)
    xgb_grid.fit(X_train, y_train)  
    print(type(xgb_grid))  
    return xgb_grid          


def save_best_model(result_path, model_name, grid, set = 'A'):
    
    """
    Save the best model into a particular folder.

    Parameters
    ----------
    X_train : 
        train features
    y_train :
        train labels
    result_path : 
        path to which the best model to be saved
    model_name : 
        specifies which model to be trained(RF, KNN or XG Boost)
    model : 
        trained model information passed as grid
    set :
        which set of values are being evaluated
    
    Returns
    -------
    Saves best model with various parameters into a file

    """

    write_params = {}
    mydir = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except:
        print('Exception raised....Check the folders')
    best_est = grid.best_estimator_    
    filename = os.path.join(mydir, 'model_name_%s_set_%s.sav'%(model_name, set))
    pickle.dump(best_est, open(filename, 'wb'))
    best_index = grid.best_index_
    best_params = grid.best_params_
    best_score = grid.best_score_
    mean_training_time = grid.cv_results_['mean_fit_time'][best_index]

    write_params['best_index'] = best_index
    write_params['best_params'] = best_params
    write_params['best_score'] = best_score
    write_params['training_time'] = mean_training_time

    df = pd.DataFrame.from_dict(write_params, orient='index')
    df.to_csv(os.path.join(mydir, 'model_name_%s_set_%s_params.csv'%(model_name, set)))

def create_matrix_for_runs(labels, labeldict_inv):
    
    """
    Create matrix of various labels.
    
    Parameters
    -----------
    labels : 
        labels for various train values.
    labeldict_inv :
        converted labels into dictionary and inversed.
    
    Returns
    -------
    matrix :
        matrix with values for different moves.
    """

    for move_1, move_2 in zip(labels, labels[1:]):
        movement_1 = labeldict_inv[move_1]
        movement_2 = labeldict_inv[move_2]
        matrix = add_key_to_matrix(movement_1, movement_2)
        
    return matrix     

def add_key_to_matrix(move_1, move_2):
    
    """
    Adding the each key to matrix if that is a particular move.
    ----------
    Parameters
    ----------
    move_1 : dict
        inversed dictionary with movement obtained from labels.
        
    move_2 : dict
        inversed dictionary with movement obtained from labels.
    -------
    Returns
    -------
    SparseMatrix : matrix
        matrix with various movements.
    """

    global SparseMatrix            
    key=move_1
    if key in SparseMatrix:
        if move_2 in SparseMatrix[key]:
            SparseMatrix[key][move_2] = SparseMatrix[key][move_2]+1
        else:
            SparseMatrix[key][move_2]=1

    else:
        if move_2 == '':
            SparseMatrix[key] = {}
        else:    
            SparseMatrix[key]={move_2:1}
        
    return SparseMatrix 

def train(result_path):

    """
    Best estimator and matrix can be obtained. 

    parameters
    ----------
    result_path : 
        path to which the training results has to be stored.
    
    Returns
    -------
    matrix : dict
        matrix with all set of movements.

    best_est : list
        best estimator for each models.
    """

    model_name = 'XG'

    json_file_path = '/Users/annarosejohny/Desktop/data_files/data+labels/annotations_set-A/20210316_r_YP27_haler_testszenario_2_set_1.json'
    json_file = read_demojson(json_file_path, demo_filename = None)

    training_features, train_labels, labeldict = run_feature_calculation([json_file])
    train_features, labels = create_feat(training_features, train_labels)

    cv_inner = KFold(n_splits=5,shuffle = True, random_state=50)

    if model_name == 'RF':
        print('Training.....')
        grid_RF = train_Rf_model(train_features, labels, cv_inner)
        
        print('Saving Model......')
        best_est =  grid_RF.best_estimator_
        save_best_model(result_path, model_name, grid_RF)

    elif model_name == 'KNN':
        print('Training......')
        grid_KNN = train_KNN_model(train_features, labels, cv_inner)
        print('Saving Model........')
        best_est =  grid_KNN.best_estimator_
        save_best_model(result_path, model_name, grid_KNN)

    else:
        print('Training......')
        grid_XGB = train_XG_Boost_model(train_features, labels, cv_inner)
        print('Saving Model.......')
        best_est =  grid_XGB.best_estimator_
        save_best_model(result_path, model_name, grid_XGB)

    labeldict_inv = {v: k for k, v in labeldict.items()}
    matrix = create_matrix_for_runs(labels, labeldict_inv)
    return matrix, best_est

def next_movement(move_1):
    """
    Finds the next movement based on current move.
    
    Parameters
    ----------
    move_1 : 
        Current move 
    
    Returns
    -------
    next_mov : 
        next movement
    """
    
    global SparseMatrix            

    key=move_1
    
    if key in SparseMatrix:
        if len(SparseMatrix[key]) > 1:
            next_mov =  max(SparseMatrix[key].items(), key=operator.itemgetter(1))[0]
        else:
            next_mov = list(SparseMatrix[key].keys())[0]
        SparseMatrix[move_1][next_mov] = SparseMatrix[move_1][next_mov]+1
    
    else:
        return 'No_Mov'
        
    return next_mov 

def test(best_est):

    """
    Estimates current accuracy and next movement accuracy based on the best estimators
    obtained from train models.
    
    Parameters
    ----------
    best_est : 
        provides best estimator based on the trained models.
    
    Returns
    -------
    curr_accuracy : int
        the current prediction accuracy.
    next_trans_acc : int
        next movement prediction accuracy.
    """
    
    json_file_path = '/Users/annarosejohny/Desktop/data_files/data+labels/annotations_set-C/20210408_r_YP27_haler_testszenario_2_set1c_only_stacking.json'
    json_file = read_demojson(json_file_path, demo_filename = None)

    testing_features, testing_labels, labeldict = run_feature_calculation([json_file])
    test_features, test_labels = create_feat(testing_features, testing_labels)
    
    curr_pred = best_est.predict(test_features[:759])
    curr_accuracy = accuracy_score(curr_pred, test_labels)
    labeldict_inv = {v: k for k, v in labeldict.items()}
    next_prediction = 0
    for idx, (test_feat) in enumerate(test_features[:-1], start = 1):
        test_feat = test_feat.reshape(1,-1)
        pred = best_est.predict(test_feat)
        pred_movement = labeldict_inv[pred[0]]
        print('curr_mov', pred_movement, end = ':::::')
        next_move = next_movement(pred_movement)
        print('next_mov', next_move)
        print('orig_mov', labeldict_inv[test_labels[idx]])
        try:
            if idx != len(test_features):
                if next_move == labeldict_inv[test_labels[idx]]:
                    next_prediction +=1
        except:
            print('End of list')
    next_trans_acc = next_prediction/(len(test_labels)-1)
    return curr_accuracy, next_trans_acc 

if __name__ == '__main__':
   
    result_path = '/Users/annarosejohny/Desktop/Haler_Future_movement_interaction/saved_models/Test_A/XG/Train_A_test_C_set1'
    matrix, best_est = train(result_path)
    curr_accuracy, future_accuracy = test(best_est)
    print('Current_accuracy', curr_accuracy, end = ':::::')
    print('Future_Accuracy', future_accuracy)
