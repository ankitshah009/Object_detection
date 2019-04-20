# coding=utf-8
# tester, given the config with model path


import tensorflow as tf
import math
import numpy as np
from utils import Dataset, grouper

class Tester():
	def __init__(self,models,config,add_mask=True):
		self.config = config
		self.models = models
		
		# infereence out:
		self.final_box_logits = [model.logits for model in models]
		

	def step(self,sess,batch):
		config = self.config
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs, batch_datas = batch
		#assert len(batch_datas) == len(self.models) # there may be less data in the end
		num_input = len(batch_datas) # use this to cap the model input

		feed_dict = {}
	
		for _,batch_data,model in zip(range(num_input), batch_datas, self.models):
			feed_dict.update(model.get_feed_dict(batch_data, is_train=False))

		sess_input = []
		
		for _, box_logit in zip(range(num_input), self.final_box_logits):
			sess_input+=[box_logit] # [K, num_class]

		outputs = sess.run(sess_input,feed_dict=feed_dict)
		pn = 1 # number of output per model
		outputs = [outputs[i*pn:(i*pn+pn)] for i in xrange(num_input)]
		return outputs

def split_batch_by_box_num(batches, box_batch_size):
	batchIdxs, batch_datas = batches
	newdata = []
	num_gpu = len(batch_datas) # each is a Dataset instance, d.data['img'] is a one item list

	num_boxes = [batch_datas[i].data['gt'][0]['boxes'].shape[0] for i in xrange(num_gpu)]
	max_num_box = max(num_boxes)
	min_num_box = min(num_boxes)

	split_into_num_batch = int(math.ceil(max_num_box/float(box_batch_size)))

	# the indexes for each inner batch
	# the batch with not enough will fill with 0, the first box
	each_batch_selected_indexes = [grouper(range(num_boxes[i]), box_batch_size, fillvalue=0) for i in xrange(num_gpu)]

	# still need to handle some batch has not enough batch
	t2 = []
	for b in each_batch_selected_indexes:
		if len(b) < split_into_num_batch:
			need = split_into_num_batch - len(b)
			b = b + [[0 for _ in xrange(box_batch_size)] for _ in xrange(need)]
		t2.append(b)

	for i in xrange(split_into_num_batch):
		this_datas = []
		for j in xrange(num_gpu):
			selected = each_batch_selected_indexes[j][i]
			temp = {
				"imgs": [batch_datas[j].data['imgs'][0]],
				"imgdata": [batch_datas[j].data['imgdata'][0]],
				"resized_image": [batch_datas[j].data['resized_image'][0]],
				'gt': [{
					"boxes": batch_datas[j].data['gt'][0]['boxes'][selected, :],
					#"labels": batch_datas[j].data['gt'][0]['labels'][selected],
				}],
			}
			this_datas.append(temp)
		newdata.append((batchIdxs, [Dataset(this_data) for this_data in this_datas]))
	return newdata

def split_data_by_box_num(data, box_batch_size):
	newdata = []
	num_gpu = len(data)

	num_boxes = [data[i]['boxes'].shape[0] for i in xrange(num_gpu)]
	max_num_box = max(num_boxes)

	split_into_num_batch = int(math.ceil(max_num_box/float(box_batch_size)))

	each_batch_selected_indexes = [grouper(range(num_boxes[i]), box_batch_size, fillvalue=0) for i in xrange(num_gpu)]

	t2 = []
	for b in each_batch_selected_indexes:
		if len(b) < split_into_num_batch:
			need = split_into_num_batch - len(b)
			b = b + [[0 for _ in xrange(box_batch_size)] for _ in xrange(need)]
		t2.append(b)

	for i in xrange(split_into_num_batch):
		this_datas = []
		for j in xrange(num_gpu):
			selected = each_batch_selected_indexes[j][i]
			
			this_datas.append({
				"image": data[j]['image'],
				"boxes": data[j]['boxes'][selected, :],
			})
		newdata.append(this_datas)
	return newdata

