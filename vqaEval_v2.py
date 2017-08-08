# coding: utf-8

import sys
dataDir = '../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools' % (dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json
import argparse

parser = argparse.ArgumentParser()

# input json
parser.add_argument('--dataType', default='mscoco', help='mscoco / abstract_v002')
parser.add_argument('--dataSubType', default='val2014', help='val2014 / train2014')
parser.add_argument('--methodInfo', default='MLB_L1', help='intermediate type')
parser.add_argument('--resultDir', default='result', help='result json folder')
parser.add_argument('--saveallfile', default=False, help='save all filetype files or not')

args = parser.parse_args()
params = vars(args)
print 'parsed input parameters:'
print json.dumps(params, indent = 2)

# set up file names and paths
dataType    =params['dataType']
dataSubType =params['dataSubType']
methodInfo  =params['methodInfo']
annFile     ='%s/Annotations/v2_%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/v2_OpenEnded_%s_%s_questions.json'%(dataDir, dataType, dataSubType)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

# An example result json file has been provided in resultDir folder.

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
	['%s/vqa_OpenEnded_%s_%s_%s_%s.json'% (params['resultDir'], dataType, dataSubType, methodInfo, fileType) \
		for fileType in fileTypes]

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
"""
print "\n"
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
"""
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Save accuracy file in: %s\n" %(accuracyFile)
json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
if params['saveallfile']:
	json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
	json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
	json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
