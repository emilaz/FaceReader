"""
.. module:: EmotionPredictor
    :synopsis: This is only for training and evaluating a classifier.
    Run after: AUvsEmotionGenerator (appends annotations)
    Run before: EmotionAppender (appends predictions)
"""

import argparse
import gc
import glob
import pickle
import dask.dataframe as dd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from multiprocessing import Pool

from dask.distributed import Client
import sys
sys.path.append('..')
from helpers.classifier_util import *
from joblib import parallel_backend


def test_classifier(df, all):
    print('Testing classifier...')
    data, labels, groups = make_emotion_data(df, all)
    classifier_path = pickle.load(open('/home/emil/facereader_classifier_test/Happy_trained_RandomForest_with_pose_emil.pkl', "rb"))
    # classifier_path = pickle.load(open('/home/emil/OpenFaceScripts/models/Happy_trained_RandomForest_with_pose_emil_good.pkl', "rb"))
    happy = get_prediction(classifier_path, data, .5)#, 0.4862)  # found during training
    #save results
    conf_mat(happy,labels,'salarian_test_conf')
    score_heatmap(happy,labels, 'salarian_test_scores')


def train_classifier(df: dd.DataFrame, path_to_save, classifier_type='rf'):
    print('Training classifier...')
    data, labels, groups = make_emotion_data(df, True)
    del df
    # split off some sessions
    gss = GroupShuffleSplit(n_splits=2, test_size=.2, random_state=4)
    tr_idx, te_idx = [(tr_te_pair) for tr_te_pair in gss.split(data, groups=groups)][0]
    X_train = data.loc[tr_idx]
    X_test = data.loc[te_idx]
    y_train = labels.loc[tr_idx]
    y_test = labels.loc[te_idx]
    group_train = groups.loc[tr_idx]
    del data
    print("TRAINING")
    rcv = make_classifier(X_train, y_train, group_train, classifier_type=classifier_type, n_iter=25)
    classifier = rcv.best_estimator_
    print('BEST CLASSIFIER:')
    print(classifier)
    if classifier_type is not 'svc':
        print('Getting optimal threshold...')
        thresh = get_optimal_threshold(classifier, X_train, y_train)
        # thresh = .5
        print('threshold:', thresh)
    else:
        thresh = None

    with open(os.path.join(path_to_save,'{}_trained.pkl'.format(classifier_type)), 'wb') as f:
        pickle.dump({'model':classifier, 'threshold':thresh, 'all_res': rcv},f)

    with parallel_backend('dask'):
        predicted = get_prediction(classifier, X_train, thresh, classifier_type)
        conf_mat(predicted, y_train, 'Confusion Matrix on Train Set')
        score_heatmap(predicted, y_train, 'Classification Metrics on Train Set')
    print('Train set classification report')
    train_rep = classification_report(y_train,predicted, output_dict=True)
    print(train_rep)

    print("PREDICTING")
    with parallel_backend('dask'):
        predicted = get_prediction(classifier, X_test, thresh, classifier_type)
        conf_mat(predicted, y_test, 'Confusion Matrix on Test Set')
        score_heatmap(predicted, y_test, 'Classification Metrics on Test Set')
        print('Test set classification report')
        test_rep = classification_report(y_test, predicted, output_dict=True)
        print(test_rep)

    total_df = pd.DataFrame(index=['Train', 'Test'], columns=['Precision', 'Recall', 'F1', 'Happy Ratio', 'Accuracy'],
                            data=[[*list(train_rep['True'].values())[:-1],
                                   '{}/{}'.format(sum(y_train), len(y_train)-sum(y_train)),
                                   train_rep['accuracy']],
                                  [*list(test_rep['True'].values())[:-1],
                                   '{}/{}'.format(sum(y_test), len(y_test)-sum(y_test)),
                                   test_rep['accuracy']]])
    total_df.to_csv(os.path.join(path_to_save,'{}_trained_metrics.csv'.format(classifier_type)))


def preprocess(df):
    print('Preprocessing...throwing away bad labels')
    data_columns = [x for x in df.columns if
                    x not in ['frame', 'face_id', 'success', 'timestamp', 'confidence', 'patient', 'session']]
    df = df[data_columns]
    data = df[df['annotated'] != "N/A"]
    data = data[data['annotated'] != ""]
    return data

def make_emotion_data(df, all=False):
    print('Making emotion data...')
    data = preprocess(df)
    if not all:
        emote_data = data[data['annotated'] == 'Happy']
        non_emote_data = data[data['annotated'] != 'Happy']
        non_emote_data = non_emote_data.sample(frac=len(emote_data) / len(non_emote_data), random_state=0)
        data = dd.concat([emote_data, non_emote_data], interleave_partitions=True)
    else:
        frac = .9
        emote_data = data[data['annotated'] == 'Happy'].sample(frac=frac, random_state=0)
        non_emote_data = data[data['annotated'] != 'Happy'].sample(frac=frac, random_state=0)
        data = dd.concat([emote_data, non_emote_data], interleave_partitions=True)
        print('WARNING. THROWING AWAY {} PERCENT CURRENTLY. DO WE WANT THAT?'.format(100*(1-frac)))
    labels = (data['annotated'] == 'Happy')
    groups = data['video']  # this is for grouping of train-test set later on
    print('HELLO DATA ANNOTATED HERE')
    del data['video']
    del data['annotated']
    data = data.compute().reset_index(drop=True)
    labels = labels.compute().reset_index(drop=True)
    groups = groups.compute().reset_index(drop=True)

    print('Created emotion data.')
    return data, labels, groups


def load_patient(patient_dir):
    try:
        curr_df = dd.read_hdf(os.path.join(patient_dir, 'hdfs', 'au_w_anno.hdf'), '/data').repartition(npartitions=200)
        curr_df = curr_df[curr_df['success'] == 1].reset_index(drop=True)

        if len(curr_df) and 'annotated' in curr_df.columns and 'frame' in curr_df.columns:
            # print(patient_dir, 'der hier hat')
            return curr_df

        else:
            print('Current df for patient {} has either no length (length {}), no annotations,'
                               'no frame column or no successful detections.'.format(patient_dir, len(curr_df)))
            return None

    except AttributeError as e:
        print(e)
    except ValueError as e:
        print(e)
    except KeyError as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="Path to OpenFaceTests directory")
    parser.add_argument("-refresh", help="Refresh DataFrame", action="store_true")
    parser.add_argument("-drop", help="Patient to leave out for training/testing")
    parser.add_argument('-classifier_type', help='Kind of classifier to train. Use rf, svc and rl.', default='rf')
    parser.add_argument("-test", help="If a classifier is just to be tested on data", action="store_true")
    args = parser.parse_args()
    if args.drop:
        leave_out = args.drop  # for leaving out a patient or a session of a patient
    else:
        leave_out = None
    au_dir = args.id
    au_path = os.path.join(au_dir,'all_aus', 'au_*.hdf')
    # if refresh, all action units are being dumped into one big dask dataframe and saved
    if args.refresh:
        if not os.path.exists(os.path.join(au_dir,'all_aus')):
            os.mkdir(os.path.join(au_dir,'all_aus'))
        else:  # if the folder already exists, make sure that it's empty to avoid having old files remain
            files_to_delete = glob.glob(os.path.join(au_dir,'all_aus/*'))
            for f in files_to_delete:
                os.remove(f)
        # this is to train only on specific patients (the ones we have crop coords for)
        pats = ['a0f66459',
        'af859cc5',
        # 'abdb496b'
         'cb46fd46',
        'd6532718',
        'e5bad52f',
        # 'd7d5f068'
        ]

        PATIENT_DIRS = [
            x for x in glob.glob('/home/emil/facereader_classifier_test/*cropped')  for p in pats if 'hdfs' in os.listdir(x)
                                                and 'au_w_anno.hdf' in os.listdir(os.path.join(x,'hdfs'))
                                                and p in x
        ]
        good_files = []  # these are all files that fall within the 7AM-23PM range
        with open('/home/emil/facereader_classifier_test/good_times.txt', 'r') as f:
            for item in f.readlines():
                good_files.append(item.rstrip())

        good_pat_dirs = [f for f in PATIENT_DIRS if f in good_files]
        pool = Pool(8)
        res = pool.map(load_patient, good_pat_dirs)
        print('filter nans')
        new_res = [r for r in res if r is not None]
        print('Good files:{}, bad files: {}'.format(len(new_res), len(res)-len(new_res)))
        print('now concat')

        df = dd.concat(new_res, interleave_partitions=True)
        gc.collect()

        df.to_hdf(au_path, '/data', format='table', scheduler='processes')  # need scheduler to avoid segfault
        print('DUMPED')
        client = Client(processes=False, silence_logs='error')
        print(client)
    else:
        client = Client(n_workers=4, threads_per_worker=2)  # , silence_logs='error')
        df = dd.read_hdf(au_path, '/data')
        # dfs = []
        # for idx,f in enumerate(glob.glob(os.path.dirname(au_path)+ '/*')):
        #     if f.endswith('.hdf'):
        #         df = pd.read_hdf(f, key='/data')
        #         data_columns = [x for x in df.columns if
        #                         x not in ['frame', 'face_id', 'success', 'timestamp', 'confidence', 'patient', 'video']]
        #         df = df[data_columns]
        #
        #         data = df[df['annotated'] != "N/A"]
        #         data = data[data['annotated'] != ""]
        #         if len(df)>0:
        #             dfs.append(df)
        #         if idx >80:
        #             break
        # df = pd.concat(dfs)
        # del dfs
        print('Files read')
    print(client)
    if args.test:
        test_classifier(df, all=False)
    else:
        train_classifier(df, au_dir, args.classifier_type)
        print('fuer den verein')
    sys.exit(0)
