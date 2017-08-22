------------------------------------------------------------------------------
--  ref [2016 arXiv] VQA: Visual Question Answering
-----------------------------------------------------------------------------

--Baseline of MLB
--Use 1x1 convolution for dimension reduction
netdef = {}
function netdef.baseline(rnn_size_q,nhimage,common_embedding_size,joint_dropout,num_layers,noutput,batch_size,glimpse)
   local p = .5  -- dropout ratio
   local activation = 'Tanh'
   local multimodal_net=nn.Sequential()
   local glimpse=glimpse or 2
   assert(num_layers==1, 'do not support stacked structure')
   print('baseline: No Shortcut')
   
   multimodal_net:add(nn.ParallelTable()
      :add(nn.Sequential()
         :add(nn.Dropout(p))
         :add(nn.Linear(rnn_size_q, common_embedding_size*glimpse))
         :add(nn[activation]()))
      :add(nn.Sequential()
         :add(nn.SpatialAveragePooling(14,14))
         :add(nn.Dropout(p))
         :add(nn.SpatialConvolution(nhimage, common_embedding_size*glimpse, 1,1))
         :add(nn[activation]()))
   ):add(nn.Sequential()
      :add(nn.CMulTable())
      :add(nn.Dropout(p))
      :add(nn.Linear(common_embedding_size*glimpse,noutput))
   )
   return multimodal_net
end

function netdef.baseline_updateBatchSize(net,nhimage,common_embedding_size,num_layers,batch_size,glimpse)
end
