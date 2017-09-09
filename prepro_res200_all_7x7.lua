------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
------------------------------------------------------------------------------

require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'
--local t = require '../fb.resnet.torch/datasets/transforms'
local t = require '../../CNN_Model/fb.resnet.torch/datasets/transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
--cmd:option('-input_json','data_train-val_test-dev_2k/data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_list_dir','.','path to the image lists')
cmd:option('-image_root','','path to the image root')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')
cmd:option('-out_path7x7', '/data/vqa/features_7x7.h5', 'path to output features')
cmd:option('-out_path7x7bn', '/data/vqa/features_7x7bn.h5', 'path to output features')
cmd:option('-out_path1x1', '/data/vqa/features_1x1.h5', 'path to output features')
cmd:option('-out_path7x7n', '/data/vqa/features_7x7n.h5', 'path to output features')
cmd:option('-out_path1x1n', '/data/vqa/features_1x1n.h5', 'path to output features')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
net=torch.load(opt.cnn_model);

net:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

print('=== Double Sized Full Crop ===')
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

local image_root = opt.image_root

-- open the mdf5 file
local features7x7 = hdf5.open(opt.out_path7x7, 'w')
local features7x7bn = hdf5.open(opt.out_path7x7bn, 'w')
local features1x1 = hdf5.open(opt.out_path1x1, 'w')
local features7x7n = hdf5.open(opt.out_path7x7n, 'w')
local features1x1n = hdf5.open(opt.out_path1x1n, 'w')

local image_list={}
local f = io.input(paths.concat(opt.image_list_dir, 'imlist_train2014.txt'))
for line in io.lines() do
    table.insert(image_list, paths.concat(image_root, 'train2014', line))
end
io.close(f)
f = io.input(paths.concat(opt.image_list_dir, 'imlist_val2014.txt'))
for line in io.lines() do
    table.insert(image_list, paths.concat(image_root, 'val2014', line))
end
io.close(f)
f = io.input(paths.concat(opt.image_list_dir, 'imlist_test2015.txt'))
for line in io.lines() do
    table.insert(image_list, paths.concat(image_root, 'test2015', line))
end
io.close(f)

local batch_size = opt.batch_size
local sz=#image_list
print(string.format('processing %d images...',sz))
local l2normalizer1x1 = nn.Normalize(2):cuda()
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(image_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat7x7=net:get(8).output:clone()
    feat7x7bn=net:get(10).output:clone()
    feat1x1=net:get(13).output:clone()
    --if opt.l2norm then
    local bs=r-i+1
    local l2normalizer=nn.Sequential()
        :add(nn.Transpose({2,3},{3,4}))
        :add(nn.Reshape(bs*7*7,2048,false))
        :add(nn.Normalize(2))
        :add(nn.Reshape(bs,7,7,2048,false))
        :add(nn.Transpose({3,4},{2,3}))
    l2normalizer=l2normalizer:cuda()
    feat7x7n=l2normalizer:forward(feat7x7)
    feat1x1n=l2normalizer1x1:forward(feat1x1)
    --end
    for j=1,r-i+1 do
        features7x7:write(paths.basename(image_list[i+j-1]), feat7x7[j]:float())
        features7x7bn:write(paths.basename(image_list[i+j-1]), feat7x7bn[j]:float())
        features1x1:write(paths.basename(image_list[i+j-1]), feat1x1[j]:float())
        features7x7n:write(paths.basename(image_list[i+j-1]), feat7x7n[j]:float())
        features1x1n:write(paths.basename(image_list[i+j-1]), feat1x1n[j]:float())
    end
    if (i-1)%50000==0 then
        collectgarbage()
    end
end

features7x7:close()
features7x7bn:close()
features1x1:close()
features7x7n:close()
features1x1n:close()
