------------------------------------------------------------------------------
--  Eval one example
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'hdf5'
require 'xlua'
require 'image'
cjson=require('cjson');
visdom = require 'visdom'
local t = require '../../CNN_Model/fb.resnet.torch/datasets/transforms'

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
cmd:option('-model_path', 'model/mrn2k.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-question', '', '')
cmd:option('-impath', '', '')
cmd:option('-l2norm', false, 'use L2-normalization')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-question_max_length', 26, 'question max length')
cmd:option('-input_encoding_size', 620, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-model_name', 'MLB', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 1, '# of layers of Multimodal Residual Networks')
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

local plot = visdom{server = 'http://localhost', port = 8097, env = 'VQAv2_test_eval_show'}
local imhdl = 'imhdl'
local vqahdl = 'vqahdl'
local attimhdl1 = 'attimhdl1'
local attimhdl2 = 'attimhdl2'
------------------------------------------------------------------------
-- Image loader
------------------------------------------------------------------------
local batch_size = 1
local nhimage=2048
local ih = 7
local iw = 7
-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
   t.Scale(224),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224)
}
imloader={}
function imloader:load(fname)
   self.im="rip"
   if not pcall(function () self.im=image.load(fname); end) then
      if not pcall(function () self.im=image.loadPNG(fname); end) then
         if not pcall(function () self.im=image.loadJPG(fname); end) then
         end
      end
   end
end
function loadim(imname)
   imloader:load(imname)
   im=imloader.im
   if im:size(1)==1 then
      im2=torch.cat(im,im,1)
      im2=torch.cat(im2,im,1)
      im=im2
   elseif im:size(1)==4 then
      im=im[{{1,3},{},{}}]
   end
   -- Scale, normalize, and crop the image
   im = transform(im)
   -- View as mini-batch of size 1
   im = im:view(1, table.unpack(im:size():totable()))
   return im
end

local l2normalizer=nn.Sequential()
   :add(nn.Transpose({2,3},{3,4}))
   :add(nn.Reshape(batch_size*ih*iw,nhimage,false))
   :add(nn.Normalize(2))
   :add(nn.Reshape(batch_size,ih,iw,nhimage,false))
   :add(nn.Transpose({3,4},{2,3}))
l2normalizer=l2normalizer:cuda()

--im=torch.CudaTensor(1,3,224,224)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
------ path setting start ------
if opt.split == 1 then input_path_prefix,opt.type = 'data_train_val','val2014'
elseif opt.split == 2 then input_path_prefix,opt.type = 'data_train-val_test','test2017'
elseif opt.split == 3 then input_path_prefix,opt.type = 'data_train-val_test-dev','test-dev2017'
end
input_path = string.format('%s_%dk', input_path_prefix, (opt.num_output/1000))
input_json = paths.concat(input_path, 'data_prepro.json')
------ path setting end ------

print('DataLoader loading json file: ', input_json)
local f = io.open(input_json, 'r')
local text = f:read()
f:close()
json_file = cjson.decode(text)
itow = json_file['ix_to_word']
itoa = json_file['ix_to_ans']
wtoi = {}
local count = 0
for i,w in pairs(itow) do
   wtoi[w] = i
   count = count + 1
end
local vocabulary_size_q=count

function string:split(sep)
   local sep, fields = sep or " ", {}
   local pattern = string.format("([^%s]+)", sep)
   self:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end

------------------------------------------------------------------------
-- Model Definitions
------------------------------------------------------------------------
-- word-embedding
local lookup = nn.LookupTableMaskZero(vocabulary_size_q, opt.input_encoding_size)
local embedding_net_q=nn.Sequential()
   :add(lookup)
   :add(nn.SplitTable(2))

-- GRU encoder
local rnn_model = nn.GRU(opt.input_encoding_size, opt.rnn_size, false, .25, true)
rnn_model:trimZero(1)
local encoder_net_q=nn.Sequential()
   :add(nn.Sequencer(rnn_model))
   :add(nn.SelectTable(-1))

require('netdef.'..opt.model_name)
local multimodal_net=netdef[opt.model_name](opt.rnn_size,nhimage,
   opt.common_embedding_size,dropout,opt.num_layers,opt.num_output,batch_size,opt.glimpse)

-- overall model
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

------------------------------------------------------------------------
-- Load VQA model
------------------------------------------------------------------------
w,dw=model:getParameters();
model_param=torch.load(opt.model_path);
w:copy(model_param);
model:evaluate();

------------------------------------------------------------------------
-- CNN model
------------------------------------------------------------------------
net=torch.load(opt.cnn_model);
net:evaluate()

function procAttentionValue(att_v)
   att_min = torch.min(att_v)
   att_max = torch.max(att_v)
   att_v = (att_v-att_min)/(att_max-att_min)
   att_v = att_v:reshape(ih,iw)
   return att_v
end

function attImage(im, att_v)
   if im:size(1)==1 then
      im2=torch.cat(im,im,1)
      im2=torch.cat(im2,im,1)
      im=im2
   elseif im:size(1)==4 then
      im=im[{{1,3},{},{}}]
   end
   oh = im:size(2)
   ow = im:size(3)
   --print(string.format('oh=%d, ow=%d',oh,ow))
   local upsampler = nn.SpatialUpSamplingBilinear({oheight=oh, owidth=ow})
   att_v = att_v:view(1, table.unpack(att_v:size():totable()))
   att2 = torch.cat(att_v,att_v,1)
   att_v = torch.cat(att2,att_v,1)
   att_v = upsampler:forward(att_v)
   return torch.cmul(im,att_v)
end

function resizeImage(im)
   if im:size(1)==1 then
      im2=torch.cat(im,im,1)
      im2=torch.cat(im2,im,1)
      im=im2
   elseif im:size(1)==4 then
      im=im[{{1,3},{},{}}]
   end
   local inh = im:size(2)
   local inw = im:size(3)
   local oh = 400
   local ow = oh/inh*inw
   local upsampler = nn.SpatialUpSamplingBilinear({oheight=oh, owidth=ow})
   im = upsampler:forward(im)
   return im
end

function process(imname, question)
   -- Image feature
   im = loadim(imname):cuda()
   net:forward(im);
   fv_im=net:get(8):get(3):get(2).output:clone()
   if opt.l2norm then
      fv_im=l2normalizer:forward(fv_im)
   end

   -- Question feature
   local ques=question:lower()
   ques=ques:gsub('n\'t', ' not')
   ques=ques:gsub('%p', ' %1 ')
   ques=ques:split()
   local lengths_q = math.min(#ques,opt.question_max_length)
   local fv_q = torch.CudaTensor(1,lengths_q)
   for i=1,lengths_q do
      fv_q[{{1},{i}}]=tonumber(wtoi[ques[i]])
   end
   fv_q = fv_q[{{},{1,lengths_q}}]
   fv_sorted_q = right_align(fv_q,torch.Tensor{lengths_q})

   -- Prediction
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
   end
   scores = model:forward({fv_sorted_q, fv_im}):double()
   tmp,pred=torch.max(scores,2);
   answer=itoa[tostring(pred[{1,1}])]
   print(string.format('Answer: %s\n', answer))

   att_v1 = model:get(2):get(1):get(3):get(8).output[1]:float() -- 7/8
   att_v2 = model:get(2):get(1):get(3):get(8).output[2]:float() -- 7/8
   att_v1 = procAttentionValue(att_v1)
   att_v2 = procAttentionValue(att_v2)

   -- Plot
   -- reload original image
   imloader:load(imname)
   showim = resizeImage(imloader.im)
   vqahdl = plot:text{
      text = string.format('Question: %s   Answer: %s', question, answer),
      win = vqahdl
   }

   attimhdl1 = plot:image{
      img = attImage(showim, att_v1),
      win = attimhdl1,
      options = { title='Attention 1' }
   }

   attimhdl2 = plot:image{
      img = attImage(showim, att_v2),
      win = attimhdl2,
      options = { title='Attention 2' }
   }
end

local imname = opt.impath
imloader:load(imname)
showim = resizeImage(imloader.im)
imhdl = plot:image{
   img = showim,
   win = imhdl,
   options = { title=imname }
}
local question=opt.question
repeat
   process(imname, question)
   io.write("Image path: ")
   io.flush()
   imname=io.read()
   imloader:load(imname)
   showim = resizeImage(imloader.im)
   imhdl = plot:image{
      img = showim,
      win = imhdl,
      options = { title=imname }
   }
   io.write("Question: ")
   io.flush()
   question=io.read()
until imname=='' or question==''