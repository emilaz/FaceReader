"""
.. module:: EmotionPredictor
    :synopsis: This is only for training and evaluating a classifier.
    Run after: AUvsEmotionGenerator (appends annotations)
    Run before: EmotionAppender (appends predictions)
"""

import argparse
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import os
import gc
import glob
import numpy as np
import pickle
import functools
from threading import Lock
from tqdm import tqdm
import time
import sys
import warnings
import dask.dataframe as dd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, make_scorer
#from sklearn.model_selection import GridSearchCV
from dask_ml.model_selection import RandomizedSearchCV

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend



# sys.path.append(
#     os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_classifier(out_q, emotion, df: dd.DataFrame):

    data, labels, groups = make_emotion_data('Happy',df)
    print(data.columns,len(groups))
    # X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(data, labels, groups,shuffle=True)
    #split off some sessions
    gss = GroupShuffleSplit(n_splits=2, test_size=.2)
    tr_idx, te_idx = [(tr_te_pair) for tr_te_pair in gss.split(data, groups=groups)][0]
    X_train = data.loc[tr_idx]
    X_test= data.loc[te_idx]
    y_train = labels.loc[tr_idx]
    y_test = labels.loc[te_idx]
    group_train = groups.loc[tr_idx]
    group_test = groups.loc[te_idx]
    scorer = make_scorer(average_precision_score,needs_proba=True)
    print("TRAINING")
    start = time.time()
    best_classifier = make_random_forest(X_train,y_train, group_train, scorer)
    stop = time.time()

    best_idx = best_classifier.best_index_
    results = best_classifier.cv_results_['mean_test_score'][best_idx]
    classifier = best_classifier.best_estimator_

    print('Getting optimal threshold...')
    thresh = get_optimal_threshold(classifier,X_train,y_train)
    print('threshold:', thresh)


    out_q.put("Best Hyperparas: {}, best threshold {}  \n".format(best_classifier.best_params_, thresh))
    out_q.put('Time it took for classifier training:{}\n'.format( stop-start))
    out_q.put('Mean Avg.Pr Score (avg. over diff. thresholds, avg. over different folds {}\n'.format(
        results
    ))

    # out_q.put("During CV (these are on the test fold)\n: Mean PR {}\n Mean REC {}\n Mean F1 {} \n".format(
    #     results['mean_test_precision'][best_idx],results['mean_test_recall'][best_idx],results['mean_test_f1'][best_idx]))
    with parallel_backend('dask'):
        predicted = get_prediction(classifier, X_train, thresh)
        out_q.put("Results on train set:\n%s\n" %
                (metrics.classification_report(y_train, predicted)))
        out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(
            y_train, predicted))
        conf_mat(predicted,y_train, 'Confusion Matrix on Train Set')
        score_heatmap(predicted,y_train,'Classification Metrics on Train Set')


    out_q.put('Training scores\n')

    print("PREDICTING")
    expected= y_test
    with parallel_backend('dask'):
        predicted = get_prediction(classifier,X_test,thresh)
        out_q.put("Results on separate test set (session seen during training):\n%s\n" %
                (metrics.classification_report(expected, predicted)))
        out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(
            expected, predicted))
        conf_mat(predicted,y_test, 'Confusion Matrix on Test Set')
        score_heatmap(predicted,y_test,'Classification Metrics on Test Set')
    pickle.dump(
        classifier,
        open('{0}_trained_RandomForest_with_pose_emil.pkl'.format(emotion), 'wb'))


def make_random_forest(feats, labels, groups, scoring):
    param_grid = {
        'n_estimators': [r for r in range(25,151,25)],
        'max_features':['auto','log2']+list(np.random.choice(np.arange(60)[1:],20)),
        'max_depth': np.random.choice(np.arange(60)[1:],10),
    }
    groupkfold = GroupKFold(n_splits=8)
    random_forest = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=15, scoring=scoring, n_jobs=multiprocessing.cpu_count(), cv=groupkfold)
    random_forest.fit(feats, labels, groups=groups)
    return random_forest



def get_optimal_threshold(classifier,x,y): #are we optimizing for f1? or tpr-fpr?
    with parallel_backend('dask'):
        probas_ = classifier.predict_proba(x)
        # Compute ROC curve
        #this returns different tpr/fpr for different decision thresholds
        pre, rec, thresholds = metrics.precision_recall_curve(y,probas_[:,1])
    f1 = get_f1_from_pr(pre,rec)
    optimal_idx = np.argmax(f1)

    return thresholds[optimal_idx]


def get_f1_from_pr(precision,recall):
    ret = np.zeros(len(precision))
    numer = 2*precision*recall
    denom = precision+recall
    well_defined = denom != 0 #if precision or recall is zero, we return zero
    if np.any(well_defined == False):
        warnings.warn('Precision and Recall were zero for at least one entry.'
                      'Setting the respective F1 scores to zero.')
    ret[well_defined]= numer[well_defined]/denom[well_defined]
    #this is not clean, but servers as sanity check for above method
    f1_arr = 2*precision*recall/np.maximum(1e-6*np.ones(len(precision)),precision+recall)
    if not np.all(ret == f1_arr):
        raise ArithmeticError('Something is wrong in the calculation of the F1 score.')
    return ret

def get_prediction(classifier,x,thresh):
    with parallel_backend('dask'):
        y_pred = (classifier.predict_proba(x)[:,1] >= thresh).astype(bool)
    return y_pred

def conf_mat(pred,true, title):
    tn,fp,fn,tp = metrics.confusion_matrix(true, pred).ravel()
    rates = np.array([tp,fp,fn,tn]).reshape((2,2))
    df_cm = pd.DataFrame(rates, index = ['Pred Happy','Pred Not Happy'],columns = ['True Happy','True Not Happy'])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True,fmt='g',annot_kws={"size": 26})
    plt.title(title)
    plt.savefig(os.path.join('/home/emil/OpenFaceScripts/images',title))
    plt.show()

def score_heatmap(pred, true, title):
    met_dict = metrics.classification_report(true, pred, output_dict=True)
    df = pd.DataFrame(met_dict)
    del df['weighted avg']
    df.drop(index ='support', inplace=True)
    acc = df['accuracy'].values
    del df['accuracy']
    print(acc)
    df.loc[len(df)] = [np.nan]*2+[acc[0]]
    df.rename(columns={'macro avg':'Total', 'False':'Not happy', 'True':'Happy'}, index={3:'accuracy'},inplace=True)
    plt.figure(figsize = (10,7))
    sns.heatmap(df.T,annot=True,cmap='Reds', fmt='g', vmin = 0,vmax=1)
    plt.yticks(rotation=0, fontsize="10", va="center")
    plt.title(title)
    plt.savefig(os.path.join('/home/emil/OpenFaceScripts/images',title))
    plt.show()



def make_emotion_data(emotion, df):
    print('Making emotion data...')
    # data_columns = [x for x in df.columns if 'predicted' not in x and 'patient' not in x and 'session' not in x and 'vid' not in x]
    data_columns = [x for x in df.columns if x not in ['frame', 'face_id', 'success', 'timestamp', 'confidence', 'patient', 'video']]
    df = df[data_columns]

    data = df[df['annotated'] != "N/A"]
    data = data[data['annotated'] != ""]
    emote_data = data[data['annotated'] == emotion]
    non_emote_data = data[data['annotated'] != emotion]

    #I think this is for balancing Emo/Non-Emo samples during training
    ####shuffle approach
    non_emote_data = non_emote_data.sample(frac=len(emote_data)/len(non_emote_data))
    data = dd.concat([emote_data, non_emote_data], interleave_partitions=True)
    labels = (data['annotated'] == emotion)
    groups = data['session']

    del data['session']
    del data['annotated']
    data = data.compute().reset_index(drop=True)
    labels = labels.compute().reset_index(drop=True)
    groups = groups.compute().reset_index(drop=True)
    ####

    ####non-shuffle approach
    # non_emote_data = non_emote_data.sample(frac=len(emote_data) / len(non_emote_data))
    # data = dd.concat([emote_data, non_emote_data], interleave_partitions=True).compute().sort_index().reset_index(
    #     drop=True)
    # labels = (data['annotated'] == emotion)
    # del data['annotated']
    ####

    print('Done')
    return data, labels, groups

def load_patient(out_q, patient_dir):
        try:
            lock = Lock()
            curr_df = dd.read_hdf(os.path.join(patient_dir, 'hdfs', 'au_w_anno.hdf'), '/data')
            curr_df = curr_df[curr_df['success'] == 1]

            if len(curr_df) and 'annotated' in curr_df.columns and 'frame' in curr_df.columns:
                out_q.put(curr_df)

        except AttributeError as e:
            print(e)
        except ValueError as e:
            print(e)
        except KeyError as e:
            print(e)

def dump_queue(queue):
    result = []

    while not queue.empty():
        i = queue.get()
        result.append(i)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="Path to OpenFaceTests directory")
    parser.add_argument("-refresh", help="Refresh DataFrame", action="store_true")
    parser.add_argument("-drop", help="Patient to leave out for training/testing")
    args = parser.parse_args()
    if args.drop:
        leave_out = args.drop
    else:
        leave_out = None
    OpenDir = args.id
    os.chdir(OpenDir)
    au_path = os.path.join('all_aus', 'au_*.hdf')
    #if refresh, all action units are being dumped into one big dask dataframe and saved
    if args.refresh:
        if not os.path.exists('all_aus'):
            os.mkdir('all_aus')
        else: #if the folder already exists, make sure that it's empty to avoid having old files remain
            files_to_delete = glob.glob('all_aus/*')
            for f in files_to_delete:
                os.remove(f)
        PATIENT_DIRS = [
            x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x) #and leave_out not in x #this is for leaving out 1 patient
        ]
        dfs = []
        df = None
        patient_queue = multiprocessing.Manager().Queue()
        partial_patient_func = functools.partial(load_patient, patient_queue)
        max_ = len(PATIENT_DIRS)
        with Pool() as p:
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap(partial_patient_func, PATIENT_DIRS[:max_], chunksize=100)):
                    pbar.update()
        ###
        #just for now:
        # dask_hdfs = dump_queue(patient_queue)
        # pd_hdfs = [d.compute() for d in dask_hdfs]
        # df = pd.concat(pd_hdfs)
        # df = dd.from_pandas(df, chunksize=5000)
        # ###
        df = dd.concat(dump_queue(patient_queue), interleave_partitions=True)
        del patient_queue
        gc.collect()

        df.to_hdf(au_path, '/data', format='table', scheduler='processes') #need scheduler to avoid segfault
        print('DUMPED')
        client = Client(processes=False)
        print(client)

    else:
        client = Client(processes=False)
        print(client)
        df = dd.read_hdf(au_path, '/data')
        print('Files read')

    out_file = open(os.path.join('all_aus','classifier_performance.txt'), 'w')
    out_q = multiprocessing.Manager().Queue()

    for emotion in ['Happy']:
        train_classifier(out_q, emotion, df)

    # print(dump_queue(out_q))

    while not out_q.empty():
        out_file.write(out_q.get())
    out_file.close()
    print('fuer den verein')
    quit()
