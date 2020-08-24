import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
# from dask_ml.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from dask.distributed import progress
import numpy as np
from sklearn import metrics
import warnings
import multiprocessing
from joblib import parallel_backend
import os
import pickle

def make_classifier(feats, labels, groups, classifier_type, n_iter = 15):
    scoring = make_scorer(average_precision_score, needs_proba=True)
    groupkfold = GroupKFold(n_splits=6)
    if classifier_type == 'rf':
        param_grid = {
            'n_estimators': [r for r in range(25, 151, 25)],
            'max_features': ['auto', 'log2'] + list(np.random.choice(np.arange(60)[1:], 20)),
            'max_depth': np.random.choice(np.arange(60)[1:], 10),
        }
        classifier = RandomForestClassifier(class_weight='balanced')
    elif classifier_type == 'svc':
        param_grid = {
            'feature_map__gamma': np.logspace(-5, -1, 4),
            'feature_map__n_components': np.arange(start=400, stop=1500),
            'svm__C': np.logspace(-1, 5, 7),
            'feature_map__degree': np.arange(2,6),
            'feature_map__kernel': ['poly']
        }
        classifier = pipeline.Pipeline([('scaler', StandardScaler()), ('feature_map', Nystroem(random_state=1)),
                                  ('svm', svm.LinearSVC(class_weight='balanced'))], verbose=True)
        scoring = 'f1'  # LinearSVC does not have predict_proba implemented
    elif classifier_type == 'lr':
        param_grid = {
            'lr__C': np.logspace(-4,4,9)
        }
        classifier = pipeline.Pipeline(
            [('scaler', StandardScaler()),
             ('lr', LogisticRegression(class_weight='balanced', solver='sag', max_iter=1e2))], verbose=True)

    else:
        raise ValueError('Classifier Type not supported. Please choose rf, svc or lr.')
    rcv = RandomizedSearchCV(classifier, param_grid, n_iter=n_iter, scoring=scoring, n_jobs=8,
                             cv=groupkfold, iid=False, verbose=2)
    with parallel_backend('threading', n_jobs=8):
        rcv.fit(feats, labels, groups=groups)
    print('Trained Classifier.')
    return rcv


def get_optimal_threshold(classifier, x, y):  # are we optimizing for f1? or tpr-fpr?
    with parallel_backend('dask'):
        probas_ = classifier.predict_proba(x)
        # Compute ROC curve
        # this returns different tpr/fpr for different decision thresholds
        pre, rec, thresholds = metrics.precision_recall_curve(y, probas_[:, 1])
    f1 = get_f1_from_pr(pre, rec)
    optimal_idx = np.argmax(f1)

    return thresholds[optimal_idx]


def get_f1_from_pr(precision, recall):
    ret = np.zeros(len(precision))
    numer = 2 * precision * recall
    denom = precision + recall
    well_defined = denom != 0  # if precision or recall is zero, we return zero
    if np.any(well_defined == False):
        warnings.warn('Precision and Recall were zero for at least one entry.'
                      'Setting the respective F1 scores to zero.')
    ret[well_defined] = numer[well_defined] / denom[well_defined]
    # this is not clean, but servers as sanity check for above method
    f1_arr = 2 * precision * recall / np.maximum(1e-6 * np.ones(len(precision)), precision + recall)
    if not np.all(ret == f1_arr):
        raise ArithmeticError('Something is wrong in the calculation of the F1 score.')
    return ret


def get_prediction(classifier, x, thresh, classifier_type):
    if classifier_type is not 'svc':
        with parallel_backend('dask'):
            y_pred = (classifier.predict_proba(x)[:, 1] >= thresh).astype(bool)
    else:
        with parallel_backend('dask'):
            y_pred = classifier.predict(x).astype(bool)
    return y_pred

def conf_mat(pred, true, title):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    rates = np.array([tp, fp, fn, tn]).reshape((2, 2))
    df_cm = pd.DataFrame(rates, index=['Pred Happy', 'Pred Not Happy'], columns=['True Happy', 'True Not Happy'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 26})
    plt.title(title)
    plt.savefig(os.path.join('/home/emil/OpenFaceScripts/images', title))
    plt.show()


def score_heatmap(pred, true, title):
    met_dict = metrics.classification_report(true, pred, output_dict=True)
    df = pd.DataFrame(met_dict)
    del df['weighted avg']
    acc = df['accuracy'].values
    del df['accuracy']
    df.loc[len(df)] = [np.nan]*2+[acc[0]]
    df.rename(columns={'macro avg':'Total', 0.0:'Not Happy', 1.0:'Happy', '0.0':'Not Happy', '1.0': 'Happy'},
              index={4:'accuracy'},inplace=True)
    plt.figure(figsize = (10,7))
    sns.heatmap((df.T).round(2),annot=True,cmap='Reds', fmt='g', vmin = 0,vmax=1, annot_kws={"size": 22})
    plt.yticks(rotation=0, fontsize="10", va="center")
    plt.title(title)
    plt.savefig(os.path.join('/home/emil/OpenFaceScripts/images', title))
    plt.show()
    plt.close()