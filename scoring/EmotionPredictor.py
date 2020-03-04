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
import pickle
import functools
from threading import Lock
from tqdm import tqdm
import sys

import dask.dataframe as dd
# from dask_ml.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from dask_ml.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client

from joblib import parallel_backend

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_classifier(out_q, emotion, df: dd.DataFrame):

    data, labels = make_emotion_data('Happy',df)

    X_train, X_test, y_train, y_test = train_test_split(data, labels)#,shuffle=False)
    scoring = ['precision', 'recall', 'f1']
    print("TRAINING")

    best_classifier = make_random_forest(X_train,y_train, scoring)
    results = best_classifier.cv_results_
    best_idx = best_classifier.best_index_

    classifier = best_classifier.best_estimator_


    out_q.put("Best Hyperparas: {}  \n".format(best_classifier.best_params_))
    out_q.put("During CV: Mean PR {}, REC {}, F1 {} (these are on the test-fold)\n".format(
        results['mean_test_precision'][best_idx],results['mean_test_recall'][best_idx],results['mean_test_f1'][best_idx]))

    print("PREDICTING")
    expected= y_test
    with parallel_backend('dask'):
        predicted = classifier.predict(X_test)

        out_q.put("Results on separate test set (not seen during training):\n%s\n" %
                (metrics.classification_report(expected, predicted)))
        out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(
            expected, predicted))
    pickle.dump(
        classifier,
        open('{0}_trained_RandomForest_with_pose_emil.pkl'.format(emotion), 'wb'))


def make_random_forest(feats, labels, scoring) -> GridSearchCV:
    param_grid = {
        'n_estimators': [r for r in range(25,151,25)],
        'max_features':['auto','log2'],
        # 'max_depth': np.random.choice(np.arange(60)[1:],5),
    }

    random_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring=scoring, n_jobs=multiprocessing.cpu_count(), cv=5, refit='f1')
    random_forest.fit(feats, labels)
    return random_forest

def make_emotion_data(emotion, df):
    print('Making emotion data...')
    data_columns = [x for x in df.columns if 'predicted' not in x and 'patient' not in x and 'session' not in x and 'vid' not in x]
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

    del data['annotated']
    data = data.compute()
    labels = labels.compute()
    ####

    ####non-shuffle approach
    # non_emote_data = non_emote_data.sample(frac=len(emote_data) / len(non_emote_data))
    # data = dd.concat([emote_data, non_emote_data], interleave_partitions=True).compute().sort_index().reset_index(
    #     drop=True)
    # labels = (data['annotated'] == emotion)
    # del data['annotated']
    ####

    print('Done')
    return data, labels

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
    leave_out = args.drop
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
            x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x) and leave_out not in x #this is for leaving out 1 patient
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
        lock = Lock()
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
