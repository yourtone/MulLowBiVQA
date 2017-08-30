------------------------------------------------------------------------------
--  ref [2016 arXiv] VQA: Visual Question Answering
-----------------------------------------------------------------------------

--attention of MLB
--Use 1x1 convolution for dimension reduction
netdef = {}
function netdef.attention(rnn_size_q,nhimage,common_embedding_size,joint_dropout,num_layers,noutput,batch_size,glimpse)
   local p = joint_dropout  -- dropout ratio
   local activation = 'Tanh'
   local multimodal_net=nn.Sequential()
   local glimpse=glimpse or 2
   local iw=14/2
   local ih=14/2
   assert(num_layers==1, 'do not support stacked structure')
   print('attention')
   
   local attention=nn.Sequential()  -- attention networks
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.Dropout(p))
            :add(nn.Linear(rnn_size_q, common_embedding_size))
            :add(nn[activation]())
            :add(nn.Replicate(iw*ih,1,1)))
         :add(nn.Sequential()
            :add(nn.Dropout(p))
            :add(nn.SpatialConvolution(nhimage,common_embedding_size,1,1))
            :add(nn[activation]())
            :add(nn.Transpose({1,2},{2,3}):setNumInputDims(3))
            :add(nn.Reshape(iw*ih, common_embedding_size, true))))
      :add(nn.CMulTable())
      :add(nn.Reshape(iw, ih, common_embedding_size, true))
      :add(nn.Transpose({2,3},{1,2}):setNumInputDims(3))
      :add(nn.SpatialConvolution(common_embedding_size,glimpse,1,1,1,1))
      :add(nn.Reshape(glimpse, iw*ih, true))
      :add(nn.SplitTable(1,2))

   local para_softmax=nn.ParallelTable()
   for j=1,glimpse do
      para_softmax:add(nn.SoftMax())
   end
   attention:add(para_softmax)

   local glimpses=nn.ConcatTable()
   for i=1,glimpse do
      local visual_embedding_=nn.Sequential()
         :add(nn.ConcatTable()
            :add(nn.SelectTable(2+i))   -- softmax [3~]
            :add(nn.SelectTable(2)))  -- v
         :add(nn.ParallelTable()
            :add(nn.Identity())
            :add(nn.SplitTable(1,2)))
         :add(nn.MixtureTable())
         :add(nn.Dropout(p))
         :add(nn.Linear(nhimage, common_embedding_size))
         :add(nn[activation]())
      glimpses:add(visual_embedding_)
   end

   local visual_embedding=nn.Sequential()
      :add(glimpses)
      :add(nn.JoinTable(2))

   multimodal_net:add(nn.ConcatTable()
      :add(nn.SelectTable(1)) -- q
      :add(nn.Sequential()
         :add(nn.SelectTable(2)) -- v
         :add(nn.Transpose({1,2},{2,3}):setNumInputDims(3))
         :add(nn.Reshape(iw*ih,nhimage,true)))
      :add(attention)
   ):add(nn.FlattenTable()
   ):add(nn.ConcatTable()
      :add(nn.Sequential()
         :add(nn.SelectTable(1)) -- q
         :add(nn.Dropout(p))
         :add(nn.Linear(rnn_size_q, common_embedding_size*glimpse))
         :add(nn[activation]()))
      :add(visual_embedding)
   ):add(nn.CMulTable()
   ):add(nn.Dropout(p)
   ):add(nn.Linear(common_embedding_size*glimpse,noutput))
   return multimodal_net
end
