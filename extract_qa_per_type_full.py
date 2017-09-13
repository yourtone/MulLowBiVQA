# coding: utf-8

import sys
dataDir = '../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
import json
import argparse
import h5py
import numpy as np

# Extract QT per 65 qtypes
# Answers are represented by answer_word
# Include 'allA_test' and 'uniqA_test'

def main(params):
    # set up file names and paths
    dataSubType = 'train2014'
    annFile     = '%s/Annotations/v2_mscoco_%s_annotations.json'%(dataDir, dataSubType)
    quesFile    = '%s/Questions/v2_OpenEnded_mscoco_%s_questions.json'%(dataDir, dataSubType)
    train_vqa   = VQA(annFile, quesFile)

    dataSubType = 'val2014'
    annFile     = '%s/Annotations/v2_mscoco_%s_annotations.json'%(dataDir, dataSubType)
    quesFile    = '%s/Questions/v2_OpenEnded_mscoco_%s_questions.json'%(dataDir, dataSubType)
    test_vqa    = VQA(annFile, quesFile)

    ###### load QA type vocab - itoq ######
    fname_qavocab = 'data/vqa_qatype_s%d_vocab.json'%params['split']
    qavocab = json.load(open(fname_qavocab,'r'))
    itoq = qavocab['itoq'] # [qtypeid -> qtypename]
    N = len(itoq)
    #=== create Qtype object ===#
    QT = {} # [1,65]
    for i in range(N):
        QT[i+1] = {}
        QT[i+1]['name'] = itoq[str(i+1)]
        QT[i+1]['allQid_train'] = []
        QT[i+1]['allA_train'] = []
        QT[i+1]['allQid_test'] = []
        QT[i+1]['allA_test'] = []

    ###### load Q types - [qid -> qtypeid] ######
    fname_qtype_train = 'data/vqa_qtype_s1_train.json'
    fname_qtype_test = 'data/vqa_qtype_s1_test.json'
    qtype_train_map = json.load(open(fname_qtype_train,'r')) # [qid -> qtypeid]
    qtype_test_map = json.load(open(fname_qtype_test,'r')) # [qid -> qtypeid]

    ###### load QA data ######
    if params['split'] == 1:
        qa_dir = 'data_train_val'
    elif params['split'] == 2:
        qa_dir = 'data_train-val_test'
    elif params['split'] == 3:
        qa_dir = 'data_train-val_test-dev'
    qa_dir = qa_dir + '_%dk'%(params['num_ans']/1000)
    input_name = '%s/data_prepro'%qa_dir
    input_h5 = input_name+'.h5'
    f = h5py.File(input_h5, 'r')
    # train
    print 'load training QA data...'
    qidhandle_train = f['question_id_train']
    ans_train = f['answers']
    N_train = len(qidhandle_train)
    qid_train = np.zeros(N_train, dtype='uint32')
    qtype_train = np.zeros(N_train, dtype='uint32')
    for i in range(N_train):
        qid_train[i] = qidhandle_train[i]
        qtype_train[i] = qtype_train_map[str(qid_train[i])] # [1,65]
        QT[qtype_train[i]]['allQid_train'].append(int(qid_train[i]))
        QT[qtype_train[i]]['allA_train'].append(train_vqa.qa[qid_train[i]]['multiple_choice_answer'])
    # test
    print 'load testing QA data...'
    qidhandle_test = f['question_id_test']
    N_test = len(qidhandle_test)
    qid_test = np.zeros(N_test, dtype='uint32')
    qtype_test = np.zeros(N_test, dtype='uint32')
    for i in range(N_test):
        qid_test[i] = qidhandle_test[i]
        qtype_test[i] = qtype_test_map[str(qid_test[i])] # [1,65]
        QT[qtype_test[i]]['allQid_test'].append(int(qid_test[i]))
        QT[qtype_test[i]]['allA_test'].append(test_vqa.qa[qid_test[i]]['multiple_choice_answer'])
    f.close()
    # analysis
    for i in range(N):
        setAnsTrain = set(QT[i+1]['allA_train'])
        setAnsTest = set(QT[i+1]['allA_test'])
        QT[i+1]['uniqA_train'] = list(setAnsTrain)
        QT[i+1]['uniqA_test'] = list(setAnsTest)
        print 'QType (%d): %s, #QA: (%d/%d), #uniqAns: (%d/%d), #overlapAns: %d, #onlyAns: (%d/%d)'% \
            (i+1, QT[i+1]['name'], \
            len(QT[i+1]['allQid_train']), len(QT[i+1]['allQid_test']), \
            len(QT[i+1]['uniqA_train']), len(QT[i+1]['uniqA_test']), \
            len(setAnsTrain&setAnsTest), \
            len(setAnsTrain-setAnsTest), len(setAnsTest-setAnsTrain))

    ###### save QA data ######
    out_name = '%s/data_pertype_full'%qa_dir
    out_h5 = out_name+'.h5'
    f = h5py.File(out_h5, 'w')
    f.create_dataset('qid_train', dtype='uint32', data=qid_train)
    f.create_dataset('qtype_train', dtype='uint32', data=qtype_train)
    f.create_dataset('qid_test', dtype='uint32', data=qid_test)
    f.create_dataset('qtype_test', dtype='uint32', data=qtype_test)
    f.close()
    print 'wrote ', out_h5

    # create output json file
    out_json = out_name+'.json'
    json.dump(QT, open(out_json, 'w')) # each entry[qtypeid] with name, allQid_(train|test), allA_train, uniqA_train
    print 'wrote ', out_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1, type=int, help='1: Train|Val, 2: Train+Val|Test, 3: Train+Val|Test-dev')
    parser.add_argument('--num_ans', default=2000, type=int, help='number of top answers')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)