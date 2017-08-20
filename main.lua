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
cmd:option('-rnn_model', 'GRU', 'question embedding model')
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
cmd:option('-type', 'val2014', 'train2014|val2014|test-dev2017|test2017')

opt = cmd:parse(arg)
opt.iterPerEpoch = 240000 / opt.batch_size
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
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local num_layers = opt.num_layers
local model_path = opt.checkpoint_path
local batch_size = opt.batch_size
local nhimage = 2048
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dropout=opt.dropout
local glimpse=opt.glimpse
local decay_factor = 0.99997592083  -- math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local question_max_length=26
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
trainset['question'] = right_align(trainset['question'],trainset['lengths_q'])
testset['question'] = right_align(testset['question'],testset['lengths_q'])
trainset.N = trainset['question']:size(1)
trainset.s = 1 -- start index
trainset.ep = 1 -- start epoch
testset.N = testset['question']:size(1)

print('DataLoader loading img file: ', opt.input_img_h5)
local h5f = hdf5.open(opt.input_img_h5, 'r')
local h5_cache = mhdf5(h5f, {2048,14,14}, opt.mhdf5_size)  -- consumes 48Gb memory

collectgarbage()

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

if opt.rnn_model == 'GRU' then
   -- skip-thought vectors
   lookup = torch.load(paths.concat(input_path, 'skipthoughts.t7'))
   assert(lookup.weight:size(1)==vocabulary_size_q+1)  -- +1 for zero
   assert(lookup.weight:size(2)==embedding_size_q)
   -- Bayesian GRUs have right dropouts
   rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
   rnn_model:trimZero(1)
   --encoder: RNN body
   encoder_net_q=nn.Sequential()
               :add(nn.Sequencer(rnn_model))
               :add(nn.SelectTable(question_max_length))
end

collectgarbage()
--embedding: word-embedding
embedding_net_q=nn.Sequential()
               :add(lookup)
               :add(nn.SplitTable(2))

require('netdef.'..opt.model_name)
multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)
print(multimodal_net)

local model = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.Sequential()
         :add(embedding_net_q)
         :add(encoder_net_q))
      :add(nn.Identity()))
   :add(multimodal_net)

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   model = model:cuda()
   criterion = criterion:cuda()
end

local multimodal_w=multimodal_net:getParameters()
multimodal_w:uniform(-0.08, 0.08)
w,dw=model:getParameters()

if paths.filep(opt.load_checkpoint_path) then
   print('loading checkpoint model...')
   model_param=torch.load(opt.load_checkpoint_path)
   w:copy(model_param)
end

-- optimization parameter
local optimize={}
optimize.maxIter=opt.max_iters
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1
optimize.winit=w
print('nParams =',optimize.winit:size(1))
print('decay_factor =', decay_factor)

------------------------------------------------------------------------
-- Next batch for train/test
------------------------------------------------------------------------
function trainset:next_batch_train(batch_size)
   local s=trainset.s
   local e=math.min(s+batch_size-1,trainset.N)
   local train_bs=e-s+1
   local qinds=torch.LongTensor(train_bs):fill(0)
   local iminds=torch.LongTensor(train_bs):fill(0)
   local fv_im=torch.Tensor(train_bs,2048,14,14)
   for i=1,train_bs do
      qinds[i]=s+i-1
      iminds[i]=trainset['img_list'][qinds[i]]
      fv_im[i]:copy(h5_cache:get(paths.basename(train_list[iminds[i]])))
   end
   local fv_sorted_q=trainset['question']:index(1,qinds)
   local labels=trainset['answers']:index(1,qinds)
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda()
      labels = labels:cuda()
      fv_im = fv_im:cuda()
   end
   trainset.s = e + 1
   return fv_sorted_q,fv_im,labels,train_bs
end
function testset:next_batch_test(s,e)
   local test_bs=e-s+1
   local qinds=torch.LongTensor(test_bs):fill(0)
   local iminds=torch.LongTensor(test_bs):fill(0)
   local fv_im=torch.Tensor(test_bs,2048,14,14)
   for i=1,test_bs do
      qinds[i]=s+i-1
      iminds[i]=testset['img_list'][qinds[i]]
      fv_im[i]:copy(h5_cache:get(paths.basename(test_list[iminds[i]])))
   end
   local fv_sorted_q=testset['question']:index(1,qinds)
   local qids=testset['ques_id']:index(1,qinds)
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda()
      fv_im = fv_im:cuda()
   end
   return fv_sorted_q,fv_im,qids,test_bs
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
function JdJ(x)
   --clear gradients--
   dw:zero()
   --grab a batch--
   local fv_sorted_q,fv_im,labels,train_bs=trainset:next_batch_train(batch_size)

   if train_bs ~= batch_size then
      netdef[opt.model_name..'_updateBatchSize'](multimodal_net,nhimage,common_embedding_size,
         num_layers,train_bs,glimpse)
      model:cuda()
   end
   local scores = model:forward({fv_sorted_q, fv_im})
   local f=criterion:forward(scores,labels)
   local dscores=criterion:backward(scores,labels)
   model:backward(fv_sorted_q, dscores)
   if train_bs ~= batch_size then
      netdef[opt.model_name..'_updateBatchSize'](multimodal_net,nhimage,common_embedding_size,
         num_layers,batch_size,glimpse)
      model:cuda()
   end
   -- finish 1 epoch
   if trainset.s>trainset.N then
      trainset.ep = trainset.ep + 1
      -- for next epoch
      trainset.s = 1
      -- shuffle trainset samples
      local qinds=torch.randperm(trainset.N):long()
      trainset['question'] = trainset['question']:index(1,qinds)
      trainset['ques_id'] = trainset['ques_id']:index(1,qinds)
      trainset['lengths_q'] = trainset['lengths_q']:index(1,qinds)
      trainset['img_list'] = trainset['img_list']:index(1,qinds)
      trainset['answers'] = trainset['answers']:index(1,qinds)
      assert(trainset.N == trainset['question']:size(1))
   end

   gradients=dw
   if opt.clipping > 0 then gradients:clamp(-opt.clipping,opt.clipping) end
   if running_avg_train == nil then running_avg_train = f end
   running_avg_train=running_avg_train*0.95+f*0.05
   return f,gradients
end

------------------------------------------------------------------------
-- Testing
------------------------------------------------------------------------
function forward(s,e)
   --grab a batch--
   local fv_sorted_q,fv_im,qids,test_bs=testset:next_batch_test(s,e)

   if test_bs ~= batch_size then
      netdef[opt.model_name..'_updateBatchSize'](multimodal_net,nhimage,common_embedding_size,
         num_layers,test_bs,glimpse)
      model:cuda()
   end
   local scores = model:forward({fv_sorted_q, fv_im})
   if test_bs ~= batch_size then
      netdef[opt.model_name..'_updateBatchSize'](multimodal_net,nhimage,common_embedding_size,
         num_layers,batch_size,glimpse)
      model:cuda()
   end
   return scores:double(),qids
end

function writeAll(file,data)
   local f = io.open(file, "w")
   f:write(data)
   f:close()
end

function test(model_append)
   model:evaluate()

   local N = testset.N
   scores=torch.Tensor(N,noutput)
   qids=torch.LongTensor(N)
   for i=1,N,batch_size do
      xlua.progress(i, N)
      if batch_size>N-i then xlua.progress(N, N) end
      r=math.min(i+batch_size-1,N)
      scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r)
   end
   tmp,pred=torch.max(scores,2)
   response={}
   for i=1,N do
      table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
   end
   local oe_txt = cjson.encode(response)
   local fname = string.format('%s/vqa_OpenEnded_mscoco_%s_%s_results.json', 
      opt.out_path,opt.type,model_name..model_append)
   paths.mkdir(opt.out_path)
   writeAll(fname,oe_txt)
   collectgarbage()
   os.execute(string.format('python vqaEval_v2.py --resultDir %s --methodInfo %s > /dev/null', 
      opt.out_path,model_name..model_append))
   local f = io.open(string.format('%s/vqa_OpenEnded_mscoco_%s_%s_accuracy.json', 
      opt.out_path,opt.type,model_name..model_append), 'r')
   local text = f:read()
   f:close()
   json_acc = cjson.decode(text)

   model:training()
   return json_acc['overall']
end

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
local state={}
--local max_epochs = math.floor(opt.max_iters/4000)
local max_epochs = math.floor(opt.max_iters*batch_size/trainset.N)
local epoch = trainset.ep
local trainlosshandle, trainacchandle, testlosshandle, testacchandle
local trainlosshist = torch.DoubleTensor(opt.max_iters):fill(0)
local trainacchist  = torch.DoubleTensor(opt.max_iters):fill(0)
local testlosshist  = torch.DoubleTensor(max_epochs):fill(0)
local testacchist   = torch.DoubleTensor(max_epochs):fill(0)
local plot = visdom{server = 'http://localhost', port = 8097, env = 'VQA'}
local timer = torch.Timer()
optimize.learningRate=optimize.learningRate*decay_factor^opt.previous_iters
optimize.learningRate=optimize.learningRate*2^math.min(2, math.floor(opt.previous_iters/opt.kick_interval))
for iter = opt.previous_iters + 1, opt.max_iters do
   if iter%opt.save_checkpoint_every == 0 then
      paths.mkdir(model_path..'/save')
      torch.save(string.format('%s/save/%s_iter%d.t7',model_path,model_name,iter),w)
   end
   -- double learning rate at two iteration points
   if iter==opt.kick_interval or iter==opt.kick_interval*2 then
      optimize.learningRate=optimize.learningRate*2
      print('learining rate:', optimize.learningRate)
   end
   if opt.previous_iters == iter-1 then print('learining rate:', optimize.learningRate) end
   optim[opt.optimizer](JdJ, optimize.winit, optimize, state)
   -- draw train loss every 10 iters
   trainlosshist[iter] = running_avg_train
   if iter%10 == 0 and iter>1 then
      trainlosshandle = plot:line{
         X = torch.range(1,iter):double(),
         Y = trainlosshist:narrow(1,1,iter),
         win = trainlosshandle,
         options={markers=false, xlabel='iteration', ylabel='loss', title='Training loss'}
      }  -- create new plot if it does not yet exist, otherwise, update plot
   end
   -- print loss information every 100 iters
   if iter%100 == 0 then
      print(string.format('training loss: %f on iter: %d/%d on epoch: %d/%d',
         running_avg_train, iter, opt.max_iters, epoch, max_epochs))
   end
   if trainset.ep > epoch then -- finished one epoch
      print(string.format('====== training time per epoch: %dm%ds',timer:time().real/60,timer:time().real%60))
      -- do evaluation
      timer:reset();
      local acc = test(string.format('_iter%d',iter))
      print(string.format('====== testing time: %dm%ds',timer:time().real/60,timer:time().real%60))
      timer:reset();
      testacchist[epoch] = acc
      if epoch > 1 then
         -- draw
         testlosshandle = plot:line{
            X = torch.range(1,epoch):double(),
            Y = testacchist:narrow(1,1,epoch),
            win = testlosshandle,
            options={markers=false, xlabel='epoch', ylabel='accuracy', title='Testing accuracy'}
         }
      end
      epoch = trainset.ep
   end

   if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
      optimize.learningRate = optimize.learningRate * decay_factor -- set the decayed rate
   end
   if iter%1 == 0 then collectgarbage() end
end

-- Saving the final model
torch.save(string.format('%s/%s.t7',model_path,model_name),w)
h5f:close()
