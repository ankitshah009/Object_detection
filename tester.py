# coding=utf-8
# tester, given the config with model path


import tensorflow as tf


class Tester():
	def __init__(self,models,config,add_mask=True):
		self.config = config
		self.models = models
		
		# infereence out:
		self.final_boxes = [model.final_boxes for model in models]
		# [R]
		self.final_labels = [model.final_labels for model in models]
		self.final_probs = [model.final_probs for model in models]

		if config.add_act:
			self.act_final_boxes = [model.act_final_boxes for model in models]
			# [R]
			self.act_final_labels = [model.act_final_labels for model in models]
			self.act_final_probs = [model.act_final_probs for model in models]

		self.add_mask = add_mask

		if add_mask:
			# [R,14,14]
			self.final_masks = [model.final_masks for model in models]



	def step(self,sess,batch):
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs,batch_datas = batch
		#assert len(batch_datas) == len(self.models) # there may be less data in the end
		num_input = len(batch_datas) # use this to cap the model input

		feed_dict = {}
	
		for _,batch_data,model in zip(range(num_input),batch_datas,self.models):
			feed_dict.update(model.get_feed_dict(batch_data,is_train=False))

		sess_input = []
		if self.add_mask:
			for _,boxes,labels,probs,masks in zip(range(num_input),self.final_boxes,self.final_labels,self.final_probs,self.final_masks):
				sess_input+=[boxes,labels,probs,masks]
		else:	
			for _,boxes,labels,probs in zip(range(num_input),self.final_boxes,self.final_labels,self.final_probs):
				sess_input+=[boxes,labels,probs]

		if self.config.add_act:
			sess_input = []
			for _,boxes,labels,probs,actboxes,actlabels,actprobs in zip(range(num_input),self.final_boxes,self.final_labels,self.final_probs,self.act_final_boxes,self.act_final_labels,self.act_final_probs):
				sess_input+=[boxes,labels,probs,actboxes,actlabels,actprobs]

		
		#final_boxes, final_probs, final_labels, final_masks = sess.run([self.final_boxes, self.final_probs, self.final_labels, self.final_masks],feed_dict=feed_dict)
		#return final_boxes, final_probs, final_labels, final_masks
		outputs = sess.run(sess_input,feed_dict=feed_dict)
		if self.add_mask:
			pn = 4
		else:
			pn = 3
		if self.config.add_act:
			pn = 6
			outputs = [outputs[i*pn:(i*pn+pn)] for i in xrange(num_input)]
		else:
			outputs = [outputs[i*pn:(i*pn+pn)] for i in xrange(num_input)]
		return outputs
