------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang
--  https://arxiv.org/abs/1610.04325
--
--  This code is based on
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/train.lua
-----------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'myutils'
visdom = require 'visdom'
cjson=require('cjson')
m = require 'manifold'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')

-- Model parameter settings
cmd:option('-question_max_length', 14, 'question max length')
cmd:option('-input_encoding_size', 620, 'the encoding size of each token in the vocabulary')
cmd:option('-num_output', 2000, 'number of output answers')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1231, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

--torch.manualSeed(opt.seed)
--torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
------ path setting start ------
if opt.split == 1 then input_path_prefix,opt.type = 'data_train_val','val2014'
elseif opt.split == 2 then input_path_prefix,opt.type = 'data_train-val_test','test2017'
elseif opt.split == 3 then input_path_prefix,opt.type = 'data_train-val_test-dev','test-dev2017'
end
input_path = string.format('%s_%dk', input_path_prefix, (opt.num_output/1000))
input_ques_h5 = paths.concat(input_path, 'data_prepro.h5')
input_json = paths.concat(input_path, 'data_prepro.json')
input_queFea_h5 = paths.concat(input_path, 'data_prepro_QFea.h5')
------ path setting end ------

print('DataLoader loading json file: ', input_json)
local f = io.open(input_json, 'r')
local text = f:read()
f:close()
json_file = cjson.decode(text)

local fname_qtype_train = 'data/vqa_qtype_s1_train.json'
local fname_qtype_test = 'data/vqa_qtype_s1_test.json'
local fname_atype_train = 'data/vqa_atype_s1_train.json'
local fname_atype_test = 'data/vqa_atype_s1_test.json'
local fname_qavocab = 'data/vqa_qatype_s1_vocab.json'

print('DataLoader loading qtype json file: ', fname_qtype_train, fname_qtype_test)
local f = io.open(fname_qtype_train, 'r')
local text = f:read()
f:close()
qtype_train = cjson.decode(text)
local f = io.open(fname_qtype_test, 'r')
local text = f:read()
f:close()
qtype_test = cjson.decode(text)

print('DataLoader loading atype json file: ', fname_atype_train, fname_atype_test)
local f = io.open(fname_atype_train, 'r')
local text = f:read()
f:close()
atype_train = cjson.decode(text)
local f = io.open(fname_atype_test, 'r')
local text = f:read()
f:close()
atype_test = cjson.decode(text)

print('DataLoader loading type vocab json file: ', fname_qavocab)
local f = io.open(fname_qavocab, 'r')
local text = f:read()
f:close()
local qavocab = cjson.decode(text)
itoq = qavocab['itoq']
itoa = qavocab['itoa']

print('DataLoader loading h5 file: ', input_ques_h5)
local h5f = hdf5.open(input_ques_h5, 'r')
trainset = {}
trainset['ques_id'] = h5f:read('/question_id_train'):all()
testset = {}
testset['ques_id'] = h5f:read('/question_id_test'):all()
h5f:close()

print('DataLoader loading qfea file: ', input_queFea_h5)
local h5f = hdf5.open(input_queFea_h5, 'r')
trainset['question'] = h5f:read('/train_question'):all()
testset['question'] = h5f:read('/test_question'):all()
h5f:close()
trainset.N = trainset['question']:size(1)
testset.N = testset['question']:size(1)

collectgarbage()

print('Process ques_type...')
local N = trainset.N
local qtypes=torch.Tensor(N)
local atypes=torch.Tensor(N)
for i=1,N do
   xlua.progress(i,N)
   qtypes[i]=qtype_train[tostring(trainset['ques_id'][i])]
   atypes[i]=atype_train[tostring(trainset['ques_id'][i])]
end
xlua.progress(N,N)
trainset['ques_type'] = qtypes
trainset['ans_type'] = atypes
local N = testset.N
local qtypes=torch.Tensor(N)
local atypes=torch.Tensor(N)
for i=1,N do
   xlua.progress(i,N)
   qtypes[i]=qtype_test[tostring(testset['ques_id'][i])]
   atypes[i]=atype_test[tostring(testset['ques_id'][i])]
end
xlua.progress(N,N)
testset['ques_type'] = qtypes
testset['ans_type'] = atypes

-- t-SNE
function tablelength(T)
   local count =0
   for _ in pairs(T) do
      count = count +1
   end
   return count
end

n = 200
train_x = m.embedding.tsne(trainset['question'][{{1,n},{}}]:double(), {dim=2, perplexity=30})
--test_x = m.embedding.tsne(testset['question'][{{1,n},{}}]:double(), {dim=2, perplexity=30})
train_yq = trainset['ques_type'][{{1,n}}]
--test_yq = testset['ques_type'][{{1,n}}]
train_ya = trainset['ans_type'][{{1,n}}]
--test_ya = testset['ans_type'][{{1,n}}]

local Nq = torch.min(torch.Tensor{tablelength(itoq), torch.max(train_yq)})
legendq = {}
for i=1,Nq do
   legendq[i] = itoq[tostring(i)]
end
local Na = torch.min(torch.Tensor{tablelength(itoa), torch.max(train_ya)})
legenda = {}
for i=1,Na do
   legenda[i] = itoa[tostring(i)]
end

plot = visdom{server = 'http://localhost', port = 8097, env = 'main'}
plot:scatter{
   X = train_x,
   Y = train_yq,
   options = {
      legend = legendq,
      markersize = 10,
   }
}
plot:scatter{
   X = train_x,
   Y = train_ya,
   options = {
      legend = legenda,
      markersize = 10,
   }
}
