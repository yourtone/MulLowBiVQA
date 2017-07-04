------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
-----------------------------------------------------------------------------

--Multimodal Low-rank Bilinear Attention Networks (MLB)
--Use 1x1 convolution for dimension reduction
netdef = {}
function netdef.MLB0(rnn_size_q,nhimage,common_embedding_size,joint_dropout,num_layers,noutput,batch_size,glimpse)
   local p = .5  -- dropout ratio
   local activation = 'Tanh'
   local multimodal_net=nn.Sequential()
   local glimpse=glimpse or 2
   assert(num_layers==1, 'do not support stacked structure')
   print('MLB0: No Shortcut')
   
   local glimpses=nn.ConcatTable()
   for i=1,glimpse do
      local visual_embedding_=nn.Sequential()
            :add(nn.ConcatTable()
               :add(nn.SelectTable(2+i))   -- softmax [3~]
               :add(nn.SelectTable(2)))  -- v
            :add(nn.ParallelTable()
               :add(nn.Identity())
               :add(nn.SplitTable(2)))
            :add(nn.MixtureTable())
            :add(nn.Dropout(p))
            :add(nn.Linear(nhimage, common_embedding_size))
            :add(nn[activation]())
      glimpses:add(visual_embedding_)
   end

   local visual_embedding=nn.Sequential()
         :add(glimpses)
         :add(nn.JoinTable(2))

   for i=1,num_layers do
      local reshaper
      if i == 1 then
         reshaper = nn.Sequential()
            :add(nn.Transpose({2,3},{3,4}))
            :add(nn.Reshape(14*14, nhimage))
      else reshaper = nn.Identity() end

      local attention=nn.Sequential()  -- attention networks
            :add(nn.ParallelTable()
               :add(nn.Sequential()
                  :add(nn.Dropout(p))
                  :add(nn.Linear(rnn_size_q, common_embedding_size))
                  :add(nn[activation]())
                  :add(nn.Replicate(14*14, 2)))
               :add(nn.Sequential()
                  :add(nn.Reshape(batch_size*14*14, nhimage, false))
                  :add(nn.Dropout(p))
                  :add(nn.Linear(nhimage, common_embedding_size))
                  :add(nn[activation]())
                  :add(nn.Reshape(batch_size, 14*14, common_embedding_size, false))))
            :add(nn.CMulTable())
            :add(nn.Reshape(batch_size, 14, 14, common_embedding_size, false))
            :add(nn.Transpose({3,4},{2,3}))
            :add(nn.SpatialConvolution(common_embedding_size,glimpse,1,1,1,1))
            :add(nn.Reshape(batch_size, glimpse, 14*14, false))
            :add(nn.SplitTable(2))

      local para_softmax=nn.ParallelTable()
      for j=1,glimpse do
         para_softmax:add(nn.SoftMax())
      end
      attention:add(para_softmax)

      multimodal_net:add(nn.ParallelTable()
         :add(nn.Identity())
         :add(reshaper)
      ):add(nn.ConcatTable()
         :add(nn.SelectTable(1))  -- q
         :add(nn.SelectTable(2))  -- v1
         :add(attention)  -- second-attention
      ):add(nn.FlattenTable()
      ):add(nn.ConcatTable()
         :add(nn.Sequential()
            :add(nn.SelectTable(1))
            :add(nn.Dropout(p))
            :add(nn.Linear(rnn_size_q, common_embedding_size*glimpse))
            :add(nn[activation]()))
         :add(visual_embedding)  -- if L > 1, do clone
      ):add(nn.CMulTable()
      )
      rnn_size_q = common_embedding_size
   end
   multimodal_net:add(nn.Dropout(p))
         :add(nn.Linear(common_embedding_size*glimpse,noutput))
   return multimodal_net
end
