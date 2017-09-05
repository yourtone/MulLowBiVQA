------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang
--  https://arxiv.org/abs/1610.04325
--
--  This code is based on
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/train.lua
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'myutils'
visdom = require 'visdom'
mhdf5=require 'misc.mhdf5'
cjson=require('cjson')

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
cmd:option('-input_img_h5','data/feature/data_real_res.h5','path to the h5file containing the image feature')
cmd:option('-mhdf5_size', 10000)

-- Model parameter settings
cmd:option('-batch_size',100,'batch_size for each iterations')
--cmd:option('-rnn_model', 'GRU', 'question embedding model')
cmd:option('-question_max_length', 26, 'question max length')
cmd:option('-input_encoding_size', 620, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-model_name', 'MLB', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 1, '# of layers of Multimodal Residual Networks')
cmd:option('-dropout', .5, 'dropout probability for joint functions')
cmd:option('-glimpse', 2, '# of glimpses')
cmd:option('-clipping', 10, 'gradient clipping')

-- Optimizer parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 100, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-max_iters', 250000, 'max number of iterations to run for ')
cmd:option('-optimizer','rmsprop','opimizer')

--check point
cmd:option('-save_checkpoint_every', 25000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model', 'folder to save checkpoints')
cmd:option('-load_checkpoint_path', '', 'path to saved checkpoint')
cmd:option('-previous_iters', 0, 'previous # of iterations to get previous learning rate')
cmd:option('-kick_interval', 50000, 'interval of kicking the learning rate as its double')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1231, 'random number generator seed to use')

-- for evaluation
cmd:option('-out_path', 'result', 'path to save output json file')
--cmd:option('-type', 'val2014', 'train2014|val2014|test-dev2017|test2017')

opt = cmd:parse(arg)
opt.iterPerEpoch = opt.max_iters / opt.batch_size
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
opt.label = string.format('%s_%s', opt.label, os.date('%Y%m%dT%H%M%S'))
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local num_layers = opt.num_layers
local model_path = opt.checkpoint_path
local batch_size = opt.batch_size
local nhimage = 2048
local iw = 14
local ih = 14
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dropout=opt.dropout
local glimpse=opt.glimpse
local decay_factor = 0.99997592083  -- math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local question_max_length=opt.question_max_length
local envname = string.format('VQAv2%s_ep',opt.label)
paths.mkdir(model_path)

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
output_ques_h5 = paths.concat(input_path, 'data_prepro_QFea.h5')
------ path setting end ------

print('DataLoader loading json file: ', input_json)
local f = io.open(input_json, 'r')
local text = f:read()
f:close()
json_file = cjson.decode(text)
local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
   table.insert(train_list, imname)
end
local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
   table.insert(test_list, imname)
end
local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

print('DataLoader loading h5 file: ', input_ques_h5)
local h5f = hdf5.open(input_ques_h5, 'r')
trainset = {}
trainset['question'] = h5f:read('/ques_train'):all()
trainset['ques_id'] = h5f:read('/question_id_train'):all()
trainset['lengths_q'] = h5f:read('/ques_length_train'):all()
trainset['img_list'] = h5f:read('/img_pos_train'):all()
trainset['answers'] = h5f:read('/answers'):all()
testset = {}
testset['question'] = h5f:read('/ques_test'):all()
testset['ques_id'] = h5f:read('/question_id_test'):all()
testset['lengths_q'] = h5f:read('/ques_length_test'):all()
testset['img_list'] = h5f:read('/img_pos_test'):all()
h5f:close()
-- START trim question to be of length question_max_length
trainset['question'] = trainset['question'][{{},{1,question_max_length}}]
testset['question'] = testset['question'][{{},{1,question_max_length}}]
trainset['lengths_q']:clamp(0,question_max_length);
testset['lengths_q']:clamp(0,question_max_length);
-- END trim question to be of length question_max_length
trainset['question'] = right_align(trainset['question'],trainset['lengths_q'])
testset['question'] = right_align(testset['question'],testset['lengths_q'])
trainset.N = trainset['question']:size(1)
testset.N = testset['question']:size(1)

collectgarbage()

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

-- word-embedding
local lookup = torch.load(paths.concat(input_path, 'skipthoughts.t7'))
assert(lookup.weight:size(1)==vocabulary_size_q+1)  -- +1 for zero
assert(lookup.weight:size(2)==embedding_size_q)
local embedding_net_q=nn.Sequential()
   :add(lookup)
   :add(nn.SplitTable(2))

-- GRU encoder
local rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
rnn_model:trimZero(1)
local encoder_net_q=nn.Sequential()
   :add(nn.Sequencer(rnn_model))
   :add(nn.SelectTable(question_max_length))
collectgarbage()

-- multimodal net
require('netdef.'..opt.model_name)
local multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)

-- overall model
local model = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.Sequential()
         :add(embedding_net_q)
         :add(encoder_net_q))
      :add(nn.SpatialAveragePooling(2,2,2,2)))
   :add(multimodal_net)
print('===[Model Architecture]===')
print(model)

if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   model = model:cuda()
end

w,dw=model:getParameters()

if paths.filep(opt.load_checkpoint_path) then
   print('loading checkpoint model...')
   model_param=torch.load(opt.load_checkpoint_path)
   w:copy(model_param)
end

collectgarbage()

------------------------------------------------------------------------
-- Next batch for train/test
------------------------------------------------------------------------
function trainset:next_batch_train(s,e)
   local train_bs=e-s+1
   local qinds=torch.LongTensor(train_bs):fill(0)
   for i=1,train_bs do
      qinds[i]=s+i-1
   end
   local fv_sorted_q=trainset['question']:index(1,qinds)
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda()
   end
   return fv_sorted_q
end
function testset:next_batch_test(s,e)
   local test_bs=e-s+1
   local qinds=torch.LongTensor(test_bs):fill(0)
   for i=1,test_bs do
      qinds[i]=s+i-1
   end
   local fv_sorted_q=testset['question']:index(1,qinds)
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda()
   end
   return fv_sorted_q
end

------------------------------------------------------------------------
-- forward
------------------------------------------------------------------------
q_encoding_net = model:get(1):get(1)

function train()
   local N = trainset.N
   local qfeas=torch.Tensor(N,rnn_size_q)
   if opt.gpuid >= 0 then
      qfeas = qfeas:cuda()
   end
   for i=1,N,batch_size do
      xlua.progress(i,N)
      if batch_size>N-i then xlua.progress(N, N) end
      r=math.min(i+batch_size-1,N)
      local fv_sorted_q=trainset:next_batch_train(i,r)
      qfeas[{{i,r},{}}]=q_encoding_net:forward(fv_sorted_q)
   end
   return qfeas
end

function test()
   local N = testset.N
   local qfeas=torch.Tensor(N,rnn_size_q)
   if opt.gpuid >= 0 then
      qfeas = qfeas:cuda()
   end
   for i=1,N,batch_size do
      xlua.progress(i,N)
      if batch_size>N-i then xlua.progress(N, N) end
      r=math.min(i+batch_size-1,N)
      local fv_sorted_q=testset:next_batch_test(i,r)
      qfeas[{{i,r},{}}]=q_encoding_net:forward(fv_sorted_q)
   end
   return qfeas
end

print('DataWriter h5 file: ', output_ques_h5)
local output_qfea_h5f = hdf5.open(output_ques_h5, 'w')
output_qfea_h5f:write('train_question', train():float())
output_qfea_h5f:write('test_question', test():float())
output_qfea_h5f:close()