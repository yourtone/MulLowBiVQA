# coding: utf-8

import sys
dataDir = '../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
import json
import argparse
import h5py
import numpy as np

# Analysis test accuracy per 65 qtypes

def main(params):
    ###### load QA data ######
    if params['split'] == 1:
        qa_dir = 'data_train_val'
    elif params['split'] == 2:
        qa_dir = 'data_train-val_test'
    elif params['split'] == 3:
        qa_dir = 'data_train-val_test-dev'
    qa_dir = qa_dir + '_%dk'%(params['num_ans']/1000)

    # Load vqa object
    dataSubType = 'val2014'
    annFile     = '%s/Annotations/v2_mscoco_%s_annotations.json'%(dataDir, dataSubType)
    quesFile    = '%s/Questions/v2_OpenEnded_mscoco_%s_questions.json'%(dataDir, dataSubType)
    test_vqa    = VQA(annFile, quesFile)

    ###### load QA type vocab - itoq ######
    fname_qavocab = 'data/vqa_qatype_s%d_vocab.json'%params['split']
    qavocab = json.load(open(fname_qavocab,'r'))
    itoq = qavocab['itoq'] # [qtypeid -> qtypename]
    N = len(itoq)
    if params['qtid'] > 0:
        N = 1
    #=== create Res_Qtype object ===#
    Res = {} # [1,65]
    for i in range(N):
        Res[i+1] = {}
        Res[i+1]['name'] = itoq[str(i+1)]
        Res[i+1]['qid'] = []
        Res[i+1]['ans'] = []
        Res[i+1]['gt'] = []
        Res[i+1]['corr'] = [] # boolean
        Res[i+1]['inTrA'] = [] # boolean

    # Load qid to qtypeid mapping
    fname_qtype_test = 'data/vqa_qtype_s1_test.json'
    qtype_test_map = json.load(open(fname_qtype_test,'r')) # [qid -> qtypeid]

    # Load qtype data
    QT_file = '%s/data_pertype_full.json'%qa_dir
    QT = json.load(open(QT_file,'r'))

    results = json.load(open(params['resfile'],'r'))
    N_test = len(results)
    for i in range(N_test):
        qid = results[i]['question_id'] # array int
        ans = results[i]['answer'] # str
        qtid = qtype_test_map[str(qid)] # [1,65]
        gt = test_vqa.qa[qid]['multiple_choice_answer']
        corr = ans == gt
        inTrA = gt in QT[str(qtid)]['uniqA_train']
        if params['qtid'] > 0:
            qtid = 1
        Res[qtid]['qid'].append(int(qid))
        Res[qtid]['ans'].append(ans)
        Res[qtid]['gt'].append(gt)
        Res[qtid]['corr'].append(corr)
        Res[qtid]['inTrA'].append(inTrA)

    if params['qtid'] == 0:
        out_file = '%s/acc_analysis_pertype.json'%qa_dir
        json.dump(Res, open(out_file,'w'))
        print 'wrote ', out_file

    # analysis
    if params['qtid'] == 0:
        for i in range(N):
            corrQT = Res[i+1]['corr']
            inTrAQT = Res[i+1]['inTrA']
            outTrAQT = [not x for x in inTrAQT]
            incorr = [corrQT[j] and inTrAQT[j] for j in range(len(corrQT))]
            outcorr = [corrQT[j] and outTrAQT[j] for j in range(len(corrQT))]
            assert(len(corrQT)==sum(inTrAQT)+sum(outTrAQT))
            if sum(outTrAQT) == 0:
                print 'QType (%d): %s, acc: %.2f, inTrA acc: %.2f, outTrA acc: --'%(i+1, Res[i+1]['name'], \
                    100.*sum(corrQT)/len(corrQT), 100.*sum(incorr)/sum(inTrAQT))
            else:
                print 'QType (%d): %s, acc: %.2f, inTrA acc: %.2f, outTrA acc: %.2f'%(i+1, Res[i+1]['name'], \
                    100.*sum(corrQT)/len(corrQT), 100.*sum(incorr)/sum(inTrAQT), 100.*sum(outcorr)/sum(outTrAQT))
    else:
        corrQT = Res[1]['corr']
        inTrAQT = Res[1]['inTrA']
        outTrAQT = [not x for x in inTrAQT]
        incorr = [corrQT[j] and inTrAQT[j] for j in range(len(corrQT))]
        outcorr = [corrQT[j] and outTrAQT[j] for j in range(len(corrQT))]
        assert(len(corrQT)==sum(inTrAQT)+sum(outTrAQT))
        if sum(outTrAQT) == 0:
            print 'QType (%d): %s, acc: %.2f, inTrA acc: %.2f, outTrA acc: --'%(params['qtid'], Res[1]['name'], \
                100.*sum(corrQT)/len(corrQT), 100.*sum(incorr)/sum(inTrAQT))
        else:
            print 'QType (%d): %s, acc: %.2f, inTrA acc: %.2f, outTrA acc: %.2f'%(params['qtid'], Res[1]['name'], \
                100.*sum(corrQT)/len(corrQT), 100.*sum(incorr)/sum(inTrAQT), 100.*sum(outcorr)/sum(outTrAQT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1, type=int, help='1: Train|Val, 2: Train+Val|Test, 3: Train+Val|Test-dev')
    parser.add_argument('--num_ans', default=2000, type=int, help='number of top answers')
    parser.add_argument('--resfile', default='result/vqa_OpenEnded_mscoco_val2014_MLB_20170901T172553_L1_ep19_results.json', help='results file name')
    parser.add_argument('--qtid', default=0, type=int, help='ques type id, 0 for all types')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)