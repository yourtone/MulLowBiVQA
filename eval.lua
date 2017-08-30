------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
--
--  This code is based on 
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/eval.lua
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'hdf5'
cjson=require('cjson');
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the Visual Question Answering model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')
cmd:option('-input_img_h5','data/feature/data_real_res.h5','path to the h5file containing the image feature')
cmd:option('-model_path', 'model/mrn2k.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'result', 'path to save output json file')
cmd:option('-out_prob', false, 'save prediction probability matrix as `model_name.t7`')
--cmd:option('-type', 'val2014', 'evaluation set: train2014|val2014|test-dev2017|test2017')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-batch_size', 200,'batch_size for each iterations')
--cmd:option('-rnn_model', 'GRU', 'question embedding model')
cmd:option('-question_max_length', 26, 'question max length')
cmd:option('-input_encoding_size', 620, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-model_name', 'MLB', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 1, '# of layers of Multimodal Residual Networks')
cmd:option('-model_append','','name to append to model_name, blank for final model or _ep12 for 12 epoch model')
cmd:option('-glimpse', 2, '# of glimpses')

cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local model_path = opt.model_path
local num_layers = opt.num_layers
local batch_size = opt.batch_size
local nhimage=2048
local iw = 14
local ih = 14
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local glimpse=opt.glimpse
local question_max_length=opt.question_max_length

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
local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
   table.insert(test_list, imname)
end
local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

print('DataLoader loading h5 file: ', input_ques_h5)
local h5f = hdf5.open(input_ques_h5, 'r')
dataset = {}
dataset['question'] = h5f:read('/ques_test'):all()
dataset['ques_id'] = h5f:read('/question_id_test'):all()
dataset['lengths_q'] = h5f:read('/ques_length_test'):all()
dataset['img_list'] = h5f:read('/img_pos_test'):all()
h5f:close()
-- START trim question to be of length question_max_length
dataset['question'] = dataset['question'][{{},{1,question_max_length}}]
dataset['lengths_q']:clamp(0,question_max_length);
-- END trim question to be of length question_max_length
dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])
dataset.N = dataset['question']:size(1)

print('DataLoader loading img file: ', opt.input_img_h5)
local h5f = hdf5.open(opt.input_img_h5, 'r')

collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------

-- word-embedding
local lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
local embedding_net_q=nn.Sequential()
   :add(lookup)
   :add(nn.SplitTable(2))

-- GRU encoder
local rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
rnn_model:trimZero(1)
local encoder_net_q=nn.Sequential()
   :add(nn.Sequencer(rnn_model))
   :add(nn.SelectTable(-1))

require('netdef.'..opt.model_name)
multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)
print('===[Multimodal Architecture]===')
print(multimodal_net)

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

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   model = model:cuda()
   criterion = criterion:cuda()
end

-- setting to evaluation
model:evaluate();

w,dw=model:getParameters();
print('nParams=', w:size(1))

-- loading the model
model_param=torch.load(model_path);

-- trying to use the precedding parameters
w:copy(model_param);

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_test(s,e)
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	local fv_im=torch.Tensor(batch_size,nhimage,iw,ih);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=dataset['img_list'][qinds[i]];
		fv_im[i]:copy(h5f:read(paths.basename(test_list[iminds[i]])):all())
	end
	local fv_sorted_q=dataset['question']:index(1,qinds) 
	local qids=dataset['ques_id']:index(1,qinds);
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q=fv_sorted_q:cuda() 
		fv_im = fv_im:cuda()
	end
	return fv_sorted_q,fv_im,qids,batch_size;
end

------------------------------------------------------------------------
-- Testing
------------------------------------------------------------------------
function forward(s,e)
	--grab a batch--
	local fv_sorted_q,fv_im,qids,batch_size=dataset:next_batch_test(s,e);
	local scores = model:forward({fv_sorted_q, fv_im})
	return scores:double(),qids;
end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------
local N=dataset.N
scores=torch.Tensor(N,noutput);
qids=torch.LongTensor(N);
for i=1,N,batch_size do
	xlua.progress(i, N);if batch_size>N-i then xlua.progress(N, N) end
	r=math.min(i+batch_size-1,N);
	scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
end
h5f:close()

if opt.out_prob then torch.save(model_name..'.t7', scores); return end

tmp,pred=torch.max(scores,2);

------------------------------------------------------------------------
-- Write to json file
------------------------------------------------------------------------
function writeAll(file,data)
   local f = io.open(file, "w")
   f:write(data)
   f:close() 
end

response={};
for i=1,N do
   table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
end
local oe_txt = cjson.encode(response)
local fname = string.format('%s/vqa_OpenEnded_mscoco_%s_%s_results.json', 
   opt.out_path,opt.type,model_name..opt.model_append)
paths.mkdir(opt.out_path)
writeAll(fname,oe_txt);

if opt.split == 1 then
   os.execute(string.format('python vqaEval_v2.py --resultDir %s --methodInfo %s > /dev/null', 
      opt.out_path,model_name..opt.model_append))
   local f = io.open(string.format('%s/vqa_OpenEnded_mscoco_%s_%s_accuracy.json', 
      opt.out_path,opt.type,model_name..opt.model_append), 'r')
   local text = f:read()
   f:close()
   json_acc = cjson.decode(text)
   print('Overall accuracy: ', json_acc['overall'])
end
