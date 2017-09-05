import json
import os.path

def get_qa_types(annos):
    qtcounts = {}
    atcounts = {}
    for anno in annos:
        qtype = anno['question_type']
        atype = anno['answer_type']
        qtcounts[qtype] = qtcounts.get(qtype, 0) + 1
        atcounts[atype] = atcounts.get(atype, 0) + 1

    cnt_qtype = sorted([(count,qtype) for qtype,count in qtcounts.iteritems()], reverse=True)
    print 'top question types and their counts:'
    print '\n'.join(map(str,cnt_qtype[:10]))

    cnt_atype = sorted([(count,atype) for atype,count in atcounts.iteritems()], reverse=True)
    print 'top answer types and their counts:'
    print '\n'.join(map(str,cnt_atype[:10]))

    qtypes = []
    for i in range(len(cnt_qtype)):
        qtypes.append(cnt_qtype[i][1])

    atypes = []
    for i in range(len(cnt_atype)):
        atypes.append(cnt_atype[i][1])

    qtoi = {w:i+1 for i,w in enumerate(qtypes)}
    itoq = {i+1:w for i,w in enumerate(qtypes)}

    atoi = {w:i+1 for i,w in enumerate(atypes)}
    itoa = {i+1:w for i,w in enumerate(atypes)}

    return qtoi, itoq, atoi, itoa

def main():
    fname_train_anno = 'data/annotations/v2_mscoco_train2014_annotations.json'
    fname_test_anno = 'data/annotations/v2_mscoco_val2014_annotations.json'
    fname_qavocab = 'data/vqa_qatype_s1_vocab.json'
    fname_qtype_train = 'data/vqa_qtype_s1_train.json'
    fname_qtype_test = 'data/vqa_qtype_s1_test.json'
    fname_atype_train = 'data/vqa_atype_s1_train.json'
    fname_atype_test = 'data/vqa_atype_s1_test.json'

    train_anno = json.load(open(fname_train_anno,'r'))
    if os.path.isfile(fname_qavocab):
        out = json.load(open(fname_qavocab,'r'))
        qtoi = out['qtoi']
        itoq = out['itoq']
        atoi = out['atoi']
        itoa = out['itoa']
    else:
        qtoi, itoq, atoi, itoa = get_qa_types(train_anno['annotations'])
        out = {}
        out['qtoi'] = qtoi
        out['itoq'] = itoq
        out['atoi'] = atoi
        out['itoa'] = itoa
        json.dump(out, open(fname_qavocab, 'w'))

    #train = []
    qtype_train = {}
    atype_train = {}
    for i in range(len(train_anno['annotations'])):
        question_id = train_anno['annotations'][i]['question_id']
        question_type = train_anno['annotations'][i]['question_type']
        answer_type = train_anno['annotations'][i]['answer_type']
        #train.append({'ques_id': question_id, \
        #    'ques_type': qtoi[question_type], 'ans_type': atoi[answer_type], \
        #    'question_type': question_type, 'ans_type': answer_type})
        qtype_train[question_id] = qtoi[question_type]
        atype_train[question_id] = atoi[answer_type]

    test_anno = json.load(open(fname_test_anno,'r'))
    #test = []
    qtype_test = {}
    atype_test = {}
    for i in range(len(test_anno['annotations'])):
        question_id = test_anno['annotations'][i]['question_id']
        question_type = test_anno['annotations'][i]['question_type']
        answer_type = test_anno['annotations'][i]['answer_type']
        #test.append({'ques_id': question_id, \
        #    'ques_type': qtoi[question_type], 'ans_type': atoi[answer_type], \
        #    'question_type': question_type, 'ans_type': answer_type})
        qtype_test[question_id] = qtoi[question_type]
        atype_test[question_id] = atoi[answer_type]

    json.dump(qtype_train, open(fname_qtype_train, 'w'))
    json.dump(qtype_test, open(fname_qtype_test, 'w'))
    json.dump(atype_train, open(fname_atype_train, 'w'))
    json.dump(atype_test, open(fname_atype_test, 'w'))

if __name__ == "__main__":
    main()