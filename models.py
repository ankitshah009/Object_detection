# coding=utf-8
# model class for semantic features


import tensorflow as tf
from utils import Dataset,get_all_anchors,draw_boxes,box_wh_to_x1x2
import numpy as np
import cv2
from nn import * # yikes
import math,random,sys,os,itertools
import tensorflow.contrib.slim as slim
from nn import pretrained_resnet_conv4,conv2d,deconv2d,resnet_conv5,dense, pairwise_iou,get_iou_callable

# this is for ugly batch norm
from nn import is_training

# ------------------------------ multi gpu stuff
PS_OPS = [
	'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
	'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
	
# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(compute_device, controller_device): # ps: paramter server
	"""Returns a function to place variables on the ps_device.

	Args:
		device: Device for everything but variables
		ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

	If ps_device is not set then the variables will be placed on the default device.
	The best device for shared varibles depends on the platform as well as the
	model. Start with CPU:0 and then test GPU:0 to see if there is an
	improvement.
	"""
	def _assign(op):
		node_def = op if isinstance(op, tf.NodeDef) else op.node_def
		if node_def.op in PS_OPS:
			return controller_device
		else:
			return compute_device
	return _assign


#----------------------------------
def get_model(config,gpuid=0,task=0,controller="/cpu:0"):
	# task is not used
	#with tf.device("/gpu:%s"%gpuid):
	with tf.device(assign_to_device("/gpu:%s"%(gpuid), controller)):
		with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
			#tf.get_variable_scope().reuse_variables()
			if config.add_act:
				if config.act_v2:
					model = Mask_RCNN_FPN_Act2(config,gpuid=gpuid)
				else:
					model = Mask_RCNN_FPN_Act(config,gpuid=gpuid)
			else:
				if config.is_fpn:
					model = Mask_RCNN_FPN(config,gpuid=gpuid)
				else:
					model = Mask_RCNN(config,gpuid=gpuid)

	return model


def get_model_boxfeat(config,reuse=False):
	with tf.device("/gpu:0"):
		with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
			#tf.get_variable_scope().reuse_variables()
			model = Mask_RCNN_boxfeat(config)

	return model

# given box, get the mask
def get_model_givenbox(config,reuse=False):
	with tf.device("/gpu:0"):
		with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
			#tf.get_variable_scope().reuse_variables()
			model = Mask_RCNN_givenbox(config)

	return model

class Mask_RCNN():
	def __init__(self,config):

		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model

		self.config = config

		self.num_class = config.num_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		# for training
		self.anchor_labels = tf.placeholder(tf.int32,[None, None, config.num_anchors],name="anchor_labels")
		self.anchor_boxes = tf.placeholder(tf.float32,[None, None, config.num_anchors,4],name="anchor_boxes")
		self.gt_boxes = tf.placeholder(tf.float32,[None, 4],name="gt_boxes")
		self.gt_labels = tf.placeholder(tf.int64,[None,],name="gt_labels")

		self.gt_mask = tf.placeholder(tf.uint8,[None, None, None],name="gt_masks") # H,W,v -> {0,1}

		# the following will be added in the build_forward and loss
		self.logits = None
		self.yp = None
		self.loss = None

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		with tf.name_scope("anchors"):#,tf.device("/cpu:0"):
			fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor box coordinates for a given max_size image,
			# so for 1920 x 1920 image, 1290/16 = 120, so (120,120,NA,4) box, NA is scale*ratio boxes
			all_anchors_np = get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size) 
			all_anchors = tf.constant(all_anchors_np, name="all_anchors",dtype=tf.float32)
			# get the anchor in this image with size different than (1920,1920)
			fm_anchors = tf.slice(all_anchors, [0,0,0,0], tf.stack([fm_h,fm_w,-1,-1]),name="fm_anchors")
			self.fm_anchors = fm_anchors

		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)

			
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			p_image = p_image*(1.0/255)


			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image
		

	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		# full anchor is [120,120,num_achors,4]  for (1920,1920) image with anchor stride 16,
		fm_anchors = self.fm_anchors # [FS,FS,num_anchors,4] # all posible anchor position for this image

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		featuremap = pretrained_resnet_conv4(image, config.resnet_num_block[:3],tf_pad_reverse=config.new_tensorpack_model)

		# freeze backbone
		featuremap = tf.stop_gradient(featuremap)

		# feat -> [1,1024,image_h/16,image_w/16]
		#self.featuremap = featuremap

		# given the feature map, predict each anchor box regression (t_x,y,w,h) and label
		# [FS,FS,num_anchors,4]
		# rpn predict each anchor's regression target
		rpn_label_logits, rpn_box_logits = self.rpn_head(featuremap,1024,config.num_anchors,data_format="NCHW",scope="rpn")
		#self.rpn_box_logits = rpn_box_logits

		# [FS, FS, num_anchors, 4]
		# given the regression target t_ logits, get the real predicted box
		decoded_boxes = decode_bbox_target(rpn_box_logits, fm_anchors,decode_clip=config.bbox_decode_clip)

		# given all the predicted anchor boxes, get region proposal boxes
		# reshape box and logits to [All,4],[All]
		# do NMS since boxes are regressed from all anchors, there will be alot of overlap
		proposal_boxes, proposal_score = generate_rpn_proposals(tf.reshape(decoded_boxes, [-1,4]),tf.reshape(rpn_label_logits, [-1]), image_shape2d, config)

		#self.proposal_boxes = proposal_boxes

		if config.is_train:
			gt_boxes = self.gt_boxes
			gt_labels = self.gt_labels
			# for training, use gt_box and some proposal box as pos and neg
			# rcnn_sampled_boxes [N_FG+N_NEG,4]
			# fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
			rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(proposal_boxes, gt_boxes,gt_labels,config=config)

			proposal_boxes = rcnn_sampled_boxes
		
		# testing use all proposal boxes
		# [K,4] # training: [N_FG+N_NEG ,4]
		boxes_on_featuremap = proposal_boxes * (1.0 / config.anchor_stride)

		# given the boxes, get the fixed size features for eachbox
		# the feature is [1,1024,image_h/16,image_w/16], crop box from this feature, and then resize to a fix size as the feature for each box
		# [K,C,14,14]# training: [N_FG+N_NEG ,C,14,14]
		roi_resized = roi_align(featuremap,boxes_on_featuremap,14)

		# given the roi feature, classify each box using the fastrcnn_head
		
		# the following may fail due to zero box proposal? fixed at https://github.com/tensorflow/tensorflow/issues/14657
		"""
		# [K,1024,14,14] -> [K,2048,7,7] K = N_FG+N_NEG on training
		feature_fastrcnn = resnet_conv5(roi_resized,config.resnet_num_block[-1])
		# get roi -> prediction, regression
		# [K,num_class], [K,num_class-1,4] # training: K  = N_FG + N_NEG
		# so there should be background box, fastrcnn predict num_class and background
		# but for box regression only num_class -1
		fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_head(feature_fastrcnn,config.num_class,scope="fastrcnn")
		"""
		# tf 1.4.1 still not working # tf1.6 works
		def ff_true():
			feature_fastrcnn = resnet_conv5(roi_resized,config.resnet_num_block[-1],tf_pad_reverse=config.new_tensorpack_model)
			fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_head(feature_fastrcnn,config.num_class,scope="fastrcnn")
			return feature_fastrcnn,fastrcnn_label_logits, fastrcnn_box_logits
		def ff_false():
			ncls = config.num_class
			return tf.zeros([0, 2048, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

		feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
			tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)
		# note fastrcnn_box_logit is [K,num_class -1, 4], other is num_class

		#self.fastrcnn_box_logits = fastrcnn_box_logits

		if config.is_train:
			# rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
			fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])

			# for training, maskRCNN only apply on positive box
			#  [N_FG, 2048, 7, 7]
			fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
			# [N_FG, num_class, 14, 14]
			

			# ----------------------------------- loss
			# use input
			# anchor_boxes, anchor_labels, gt_boxes, gt_labels

			anchor_boxes = self.anchor_boxes
			anchor_labels = self.anchor_labels

			# get the gt T_xywh
			anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)

			rpn_label_loss, rpn_box_loss = self.rpn_losses(anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)

			# [N_FG, 4]
			# sampled boxes are at least iou with a gt_boxes
			fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)

			# [N_FG, 4] # each proposal box assigned gt box, may repeat
			matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

			# fastrcnn also need to regress box (just the FG box)
			encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * tf.constant(config.fastrcnn_bbox_reg_weights) #[10,10,5,5]?

			# fastrcnn input is fg and bg proposal box, do classification to num_class(include bg) and then regress on fg boxes
			# [N_FG+N_NEG,4] & [N_FG,4]
			fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_losses(rcnn_labels, fastrcnn_label_logits,encoded_boxes, tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

			# for debug
			self.rpn_label_loss = rpn_label_loss
			self.rpn_box_loss = rpn_box_loss
			self.fastrcnn_label_loss = fastrcnn_label_loss
			self.fastrcnn_box_loss = fastrcnn_box_loss

			losses = [rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss]

			# mask rcnn loss
			if config.add_mask:
				mask_logits = self.maskrcnn_head(fg_feature, config.num_class,scope='maskrcnn')
				# [N_FG, H,W]
				gt_mask = self.gt_mask
				gt_mask_for_fg = tf.gather(gt_mask,fg_inds_wrt_gt)
				# [N_FG, H, W] -> [N_FG, 14, 14]
				target_masks_for_fg = crop_and_resize(tf.expand_dims(gt_masks_for_fg,1), fg_sampled_boxes,tf.range(tf.size(fg_inds_wrt_gt)), 14)
				target_masks_for_fg = tf.squeeze(target_masks_for_fg,1)

				mrcnn_loss = self.maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)

				losses+=[mrcnn_loss]

			if config.wd is not None:
				wd = wd_cost('.*/W', config.wd,scope="wd_cost")
				losses.append(wd)

			self.loss = tf.add_n(losses,'total_loss')

			# l2loss
		else:
			# inferencing
			# K -> proposal box
			# [K,num_class]
			 
			label_probs = tf.nn.softmax(fastrcnn_label_logits)
			# get the regressed actual boxes
			# anchor box [K,4] -> [K,num_class - 1, 4] <- box regress logits [K,num_class-1,4]
			anchors = tf.tile(tf.expand_dims(proposal_boxes,1),[1, config.num_class-1,1])
			decoded_boxes = decode_bbox_target(fastrcnn_box_logits / tf.constant(config.fastrcnn_bbox_reg_weights,dtype=tf.float32), anchors)
			decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")
			#self.decoded_boxes = decoded_boxes

			# decoded boxes are [K,num_class-1,4]. so from each proposal boxes generate all classses' boxes, with prob, then do nms on these
			# pred_indices: [R,2] , each entry (#proposal[1-K], #catid [0,num_class-1])
			# final_probs [R]
			# here do nms,
			pred_indices, final_probs = self.fastrcnn_predictions(decoded_boxes, label_probs)
			# [R,4]
			final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name="final_boxes")
			# [R] , each is 1-80 catogory
			final_labels = tf.add(pred_indices[:,1],1,name="final_labels")

			if config.add_mask:
				def f1():
					# get mask prediction
					# use the final box
					roi_resized = roi_align(featuremap, final_boxes*(1.0/config.anchor_stride),14)
					feature_maskrcnn = resnet_conv5(roi_resized,config.resnet_num_block[-1],reuse=True,tf_pad_reverse=config.new_tensorpack_model)
					# [R, num_class-1, 14, 14]
					mask_logits = self.maskrcnn_head(feature_maskrcnn, config.num_class,scope='maskrcnn')
					# get only the predict class's mask
					# [num_class-1,2] -> each is (1-R, #label_class)
					indices = tf.stack([tf.range(tf.size(final_labels)),tf.to_int32(final_labels)-1],axis=1)
					# [R,14,14]
					final_mask_logits = tf.gather_nd(mask_logits,indices)
					final_masks = tf.sigmoid(final_mask_logits)
					return final_masks

				final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
				# [R,14,14]
				self.final_masks = final_masks


			# [R,4]
			self.final_boxes = final_boxes
			# [R]
			self.final_labels = final_labels
			self.final_probs = final_probs
			

	# ----some model component
	# feature map -> [1,1024,FS1,FS2] , FS1 = H/16.0, FS2 = W/16.0
	# channle -> 1024
	def rpn_head(self,featuremap, channel, num_anchors, data_format,scope="rpn"):
		with tf.variable_scope(scope):
			# [1, channel, FS1, FS2] # channel = 1024
			# conv0:W -> [3,3,1024,1024]
			h = conv2d(featuremap,channel,kernel=3,activation=tf.nn.relu,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="conv0")
			# h -> [1,1024(channel),FS1,FS2]

			# 1x1 kernel conv to classification on each grid
			# [1, 1024, FS1, FS2] -> # [1, num_anchors, FS1, FS2]
			label_logits = conv2d(h,num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="class")
			# [1, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
			box_logits = conv2d(h,4*num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="box")

			# [1,1024,FS1, FS2] -> [FS1, FS2,1024]
			label_logits = tf.squeeze(tf.transpose(label_logits, [0,2,3,1]),0)

			box_shape = tf.shape(box_logits)
			box_logits = tf.transpose(box_logits,[0,2,3,1]) # [1,FS1, FS2,1024*4]
			# [FS1, FS2,1024,4]
			box_logits = tf.reshape(box_logits,[box_shape[2], box_shape[3],num_anchors,4])

			return label_logits,box_logits

	# feature: [K,2048,7,7] # feature for each roi
	def fastrcnn_head(self,feature,num_class,scope="fastrcnn_head"):
		with tf.variable_scope(scope):
			# [K,2048,7,7] -> [K,2048]
			# global avg pooling
			feature = tf.reduce_mean(feature,axis=[2,3],name="output")
			classification = dense(feature,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
			

			if self.config.new_tensorpack_model:
				box_regression = dense(feature,num_class*4,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
				box_regression = tf.reshape(box_regression, (-1, num_class,4))

				box_regression = box_regression[:,1:,:]
				box_regression.set_shape([None,num_class-1,None])
			else:
				box_regression = dense(feature,(num_class -1)*4,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
				box_regression = tf.reshape(box_regression, (-1, num_class-1,4))
			
			

			return classification,box_regression

	
	def maskrcnn_head(self,feature,num_class,scope="maskrcnn_head"):
		with tf.variable_scope(scope):
			# feature: [K, 2048, 7, 7] # K box
			# num_class: num_cat + 1 [background]
			# return: [K, num_cat, 14, 14]
			l = deconv2d(feature, 256, kernel=2, stride=2, activation=tf.nn.relu,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_in",distribution='normal'), scope="deconv")
			l = conv2d(l,num_class-1,kernel=1,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_in",distribution='normal'), scope="conv")
			return l

	# given all proposal box prediction, based on score thres , get final NMS resulting box
	# [K,num_class-1,4] -> decoded_boxes
	# [K,num_class] label_probs
	# each proposal box has prob and box to all class
	# here using nms for each class, -> [R]
	def fastrcnn_predictions(self,boxes, probs,scope="fastrcnn_predictions"):
		with tf.variable_scope(scope):		
			config = self.config
			assert boxes.shape[1] == config.num_class - 1
			assert probs.shape[1] == config.num_class
			# transpose to map_fn along each class
			boxes = tf.transpose(boxes,[1,0,2]) # [num_class-1, K,4]
			probs = tf.transpose(probs[:,1:],[1,0]) # [num_class-1, K]

			def f(X):
				prob,box = X # [K], [K,4]
				output_shape = tf.shape(prob)
				# [K]
				ids = tf.reshape(tf.where(prob > config.result_score_thres),[-1])
				prob = tf.gather(prob,ids)
				box = tf.gather(box,ids)
				# NMS
				selection = tf.image.non_max_suppression(box,prob,max_output_size=config.result_per_im,iou_threshold=config.fastrcnn_nms_iou_thres)
				selection = tf.to_int32(tf.gather(ids,selection))
				sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

				mask = tf.sparse_to_dense(sparse_indices=sorted_selection,output_shape=output_shape,sparse_values=True,default_value=False)
				return mask

			# for each catagory get the top K
			# [num_class-1, R]
			masks = tf.map_fn(f, (probs,boxes), dtype=tf.bool, parallel_iterations=10)
			# [R,2], each entry is [cat_id,box_id]
			selected_indices = tf.where(masks)

			probs = tf.boolean_mask(probs,masks)# [num_class-1, K] 

			# topk_indices [num_class-1,result_num]
			topk_probs, topk_indices = tf.nn.top_k(probs, tf.minimum(config.result_per_im,tf.size(probs)),sorted=False)

			# [K,2] <- select [act_num,R] 
			filtered_selection = tf.gather(selected_indices, topk_indices)
			filtered_selection = tf.reverse(filtered_selection, axis=[1],name="filtered")

			# [R,2], [R,]
			return filtered_selection, topk_probs





	# ---- losses
	def maskrcnn_loss(self,mask_logits, fg_labels, fg_target_masks,scope="maskrcnn_loss"):
		with tf.variable_scope(scope):
			# mask_logits: [N_FG, num_cat, 14, 14]
			# fg_labels: [N_FG]
			# fg_target_masks: [N_FG, 14, 14]
			num_fg = tf.size(fg_labels)
			# [N_FG, 2] # these index is used to get the pos cat's logit
			indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1],axis=1)
			# ignore other class's logit
			# [N_FG, 14, 14]
			mask_logits = tf.gather_nd(mask_logits, indices)
			mask_probs = tf.sigmoid(mask_logits)

			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
			loss = tf.reduce_mean(loss, name='maskrcnn_loss')

			return loss


	def rpn_losses(self, anchor_labels, anchor_boxes, label_logits, box_logits,scope="rpn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
			# label_logits [FS,FS,num_anchors] [7,7,1024]
			# box_logits [FS,FS,num_anchors,4] [7.7,1024,4]
			
			#with tf.device("/cpu:0"):
			valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1)) # 1,0|pos/neg
			pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
			nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
			nr_pos = tf.count_nonzero(pos_mask,dtype=tf.int32, name="num_pos_anchor")

			# [K1]

			valid_anchor_labels = tf.boolean_mask(anchor_labels,valid_mask)

			# [K2]
			valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

			# label loss for all valid anchor box
			if config.focal_loss:
				label_loss = focal_loss(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))
			else:
				label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))

				label_loss = tf.reduce_mean(label_loss,name="label_loss")

			# box loss for positive anchor
			pos_anchor_boxes = tf.boolean_mask(anchor_boxes,pos_mask)
			pos_box_logits = tf.boolean_mask(box_logits,pos_mask)

			delta = 1.0/9

			# the smooth l1 loss
			box_loss = tf.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta, reduction=tf.losses.Reduction.SUM) / delta
			box_loss = tf.div(box_loss, tf.cast(nr_valid, tf.float32),name='box_loss')

			return label_loss, box_loss

	def fastrcnn_losses(self, labels, label_logits, fg_boxes, fg_box_logits,scope="fastrcnn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# label -> label for roi [N_FG + N_NEG]
			# label_logits [N_FG + N_NEG,num_class]
			# fg_boxes_logits -> [N_FG,num_class-1,4]

			# so the label is int [0-num_class], 0 being background

			if config.focal_loss:
				onehot_label = tf.one_hot(labels,label_logits.get_shape()[-1])

				# here uses sigmoid
				label_loss = focal_loss(logits=label_logits,labels=tf.to_float(onehot_label))
			else:
				label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)

				label_loss = tf.reduce_mean(label_loss, name="label_loss")

			fg_inds = tf.where(labels > 0)[:,0]
			fg_labels = tf.gather(labels, fg_inds) # [N_FG]

			num_fg = tf.size(fg_inds) # N_FG
			# [N_FG, 2]
			indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1], axis=1)
			# gather the logits from [N_FG,num_class-1, 4] to [N_FG,4], only the gt class's logit
			fg_box_logits = tf.gather_nd(fg_box_logits, indices)

			box_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

			# /  N_FG + N_NEG ?
			box_loss = tf.truediv(box_loss, tf.to_float(tf.shape(labels)[0]),name='box_loss')

			return label_loss, box_loss


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,batch,is_train=False):

		#{"imgs":[],"gt":[]}
		config = self.config
		
		N = len(batch.data['imgs'])

		assert N == 1 # only 1 image for now

		image = batch.data['imgs'][0]

		feed_dict = {}

		if batch.data.has_key("imgdata"):
			image = batch.data['imgdata'][0]
		else:
			image = cv2.imread(image,cv2.IMREAD_COLOR)
			assert image is not None,image
			image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		# resize image, boxes
		short_edge_size = config.short_edge_size
		if config.scale_jittering:
			short_edge_size = random.randint(config.short_edge_size_min,config.short_edge_size_max)
		if batch.data.has_key("resized_image"):
			resized_image = batch.data['resized_image'][0]
		else:
			resized_image = resizeImage(image,short_edge_size,config.max_size)
		newh,neww = resized_image.shape[:2]

		if is_train:
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box
			labels = anno['labels']
			assert len(labels) == len(o_boxes)

			# boxes # (x,y,w,h)
			"""
			boxes = o_boxes[:,[0,2,1,3]] #(x,w,y,h)
			boxes = boxes.reshape((-1,2,2)) #
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x,w
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y,h
			"""

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2


			# random horizontal flip
			# no flip for surveilance video?
			if config.flip_image:
				prob = 0.5
				rand = random.random()
				if rand > prob:
					resized_image = cv2.flip(resized_image,1) # 1 for horizontal
					#boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
					boxes[:,0] = neww - boxes[:,0]
					boxes[:,0,:] = boxes[:,0,::-1]# (x_min will be x_max after flip)
				

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)

			# conver box to (x1,y1,x2,y2) # updated, no need,
			#boxes[:,2] = boxes[:,0] + boxes[:,2]
			#boxes[:,3] = boxes[:,1] + boxes[:,3]

			# visualize?
			if config.vis_pre:
				label_names = [config.classId_to_class[i] for i in labels]
				o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
				boxes_x1x2 = np.asarray([box for box in boxes])
				ori_vis = draw_boxes(image,o_boxes_x1x2,labels=label_names)
				new_vis = draw_boxes(resized_image,boxes_x1x2,labels=label_names)
				imgname = os.path.splitext(os.path.basename(batch.data['imgs'][0]))[0]
				cv2.imwrite("%s.ori.jpg"%os.path.join(config.vis_path,imgname),ori_vis)
				cv2.imwrite("%s.prepro.jpg"%os.path.join(config.vis_path,imgname),new_vis)
				print "viz saved in %s"%config.vis_path
				sys.exit()

			# get rpn anchor labels
			# [fs_im,fs_im,num_anchor,4]
			try:
				fm_labels, fm_boxes = self.get_rpn_anchor_input(resized_image, boxes)
			except Exception as e: # there is no fg_rpn
				raise e

			assert len(boxes) > 0

			feed_dict[self.anchor_labels] = fm_labels
			feed_dict[self.anchor_boxes] = fm_boxes
			feed_dict[self.gt_boxes] = boxes
			feed_dict[self.gt_labels] = labels

		else:
			
			pass

		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

	def get_feed_dict_forward(self,imgdata):
		feed_dict = {}

		feed_dict[self.image] = imgdata

		feed_dict[self.is_train] = False

		return feed_dict

	# anchor related function for training--------------------

	def filter_box_inside(self, im, boxes):
		h, w = im.shape[:2]
		indices = np.where(
			(boxes[:,0] >= 0) &
			(boxes[:,1] >= 0) &
			(boxes[:,2] <= w) &
			(boxes[:,3] <= h)  
		)[0]
		return indices, boxes[indices,:]
	# for training, given image and box, get anchor box labels
	# [fs_im,fs_im,num_anchor,4] # not fs,
	def get_rpn_anchor_input(self,im,boxes):
		

		config = self.config

		boxes = boxes.copy()

		# [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
		all_anchors_np = np.copy(get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size))

		h,w = im.shape[:2]

		# TODO: change this as FPN model
		# so image may be smaller than the full anchor size
		featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
		
		# [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
		featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
		#print featuremap_anchors.shape #(46,83,15,4)
		featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))

		#anchorH, anchorW = all_anchors_np.shape[:2]
		#featureh, featurew = anchorH, anchorW
		#featuremap_anchors_flatten = all_anchors_np.reshape((-1,4))

		# num_in < FS_im*FS_im*num_anchors # [num_in,4]
		inside_ind, inside_anchors = self.filter_box_inside(im,featuremap_anchors_flatten) # the anchor box inside the image
		

		# anchor labels is in {1,-1,0}, -1 means ignore
		# N = num_in
		# [N], [N,4] # only the fg anchor has box value
		anchor_labels,anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# fill back to [fs,fs,num_anchor,4]
		# all anchor outside box is ignored (-1)

		featuremap_labels = -np.ones((featureh * featurew*config.num_anchors,),dtype='int32')
		featuremap_labels[inside_ind] = anchor_labels
		featuremap_labels = featuremap_labels.reshape((featureh,featurew,config.num_anchors))

		featuremap_boxes = np.zeros((featureh * featurew*config.num_anchors,4),dtype='float32')
		featuremap_boxes[inside_ind,:] = anchor_boxes
		featuremap_boxes = featuremap_boxes.reshape((featureh,featurew,config.num_anchors,4))

		return featuremap_labels,featuremap_boxes

	def get_anchor_labels(self,anchors,gt_boxes):
		config = self.config

		# return max_num of index for labels equal val
		def filter_box_label(labels, val, max_num):
			cur_inds = np.where(labels == val)[0]
			if len(cur_inds) > max_num:
				disable_inds = np.random.choice(cur_inds,size=(len(cur_inds) - max_num),replace=False)
				labels[disable_inds] = -1
				cur_inds = np.where(labels == val)[0]
			return cur_inds

		

		NA,NB = len(anchors),len(gt_boxes)
		assert NB > 0

		#bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
		#box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
		box_ious = np_iou(anchors, gt_boxes)

		#print box_ious.shape #(37607,7)

		#NA, each anchors max iou to any gt box, and the max gt box's index [0,NB-1]
		iou_argmax_per_anchor = box_ious.argmax(axis=1)
		iou_max_per_anchor = box_ious.max(axis=1)

		# 1 x NB, each gt box's max iou to any anchor boxes
		#iou_max_per_gt = box_ious.max(axis=1,keepdims=True) 
		#print iou_max_per_gt # all zero?
		iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

		# NA x 1? True for anchors that cover all the gt boxes
		anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

		anchor_labels = -np.ones((NA,),dtype='int32')

		anchor_labels[anchors_with_max_iou_per_gt] = 1
		anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
		anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

		# cap the number of fg anchor and bg anchor
		target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

		# set the label==1 to -1 if the number exceeds
		fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

		#assert len(fg_inds) > 0
		old_num_bg = np.sum(anchor_labels == 0)
		if old_num_bg == 0:
			raise Exception("No valid background for RPN!")

		# the rest of 256 is negative
		target_num_bg = config.rpn_batch_per_im - len(fg_inds)

		# set some label to -1 if exceeds
		filter_box_label(anchor_labels,0,target_num_bg)

		# only the fg anchor_boxes are filled with the corresponding gt_box
		anchor_boxes = np.zeros((NA,4),dtype='float32')
		anchor_boxes[fg_inds,:] = gt_boxes[iou_argmax_per_anchor[fg_inds],:]
		return anchor_labels, anchor_boxes


class Mask_RCNN_FPN():
	def __init__(self,config,gpuid=0):
		self.gpuid = gpuid
		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model

		self.config = config

		self.num_class = config.num_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		# for training
		self.anchor_labels = []
		self.anchor_boxes = []
		num_anchors = len(config.anchor_ratios)
		for k in xrange(len(config.anchor_strides)):
			self.anchor_labels.append(tf.placeholder(tf.int32,[None, None, num_anchors],name="anchor_labels_lvl%s"%(k+2)))
			self.anchor_boxes.append(tf.placeholder(tf.float32,[None, None, num_anchors,4],name="anchor_boxes_lvl%s"%(k+2)))

		self.gt_boxes = tf.placeholder(tf.float32,[None, 4],name="gt_boxes")
		self.gt_labels = tf.placeholder(tf.int64,[None,],name="gt_labels")

		self.gt_mask = tf.placeholder(tf.uint8,[None, None, None],name="gt_masks") # H,W,v -> {0,1}

		# the following will be added in the build_forward and loss
		self.logits = None
		self.yp = None
		self.loss = None

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		self.multilevel_anchors = []
		with tf.name_scope("fpn_anchors"):#,tf.device("/cpu:0"):
			#fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor box coordinates for a given max_size image,
			# so for 1920 x 1920 image, 1290/16 = 120, so (120,120,NA,4) box, NA is scale*ratio boxes
			self.multilevel_anchors = self.get_all_anchors_fpn()


		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)

			
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			p_image = p_image*(1.0/255)


			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image

	def get_all_anchors_fpn(self):
		config = self.config
		anchors = []
		assert len(config.anchor_strides) == len(config.anchor_sizes)
		for stride, size in zip(config.anchor_strides, config.anchor_sizes):
			anchors_np = get_all_anchors(stride=stride,sizes=[size],ratios=config.anchor_ratios,max_size=config.max_size) 
		
			anchors.append(anchors_np)
		return anchors
		

	def slice_feature_and_anchors(self,image_shape2d,p23456,anchors):
		# anchors is the numpy anchors for different levels
		config = self.config
		# the anchor labels and boxes are grouped into 
		gt_anchor_labels = self.anchor_labels
		gt_anchor_boxes = self.anchor_boxes
		self.sliced_anchor_labels = []
		self.sliced_anchor_boxes = []
		for i,stride in enumerate(config.anchor_strides):
			with tf.name_scope("FPN_slice_lvl%s"%(i)):
				if i<3:
					# Images are padded for p5, which are too large for p2-p4.
					pi = p23456[i]
					target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * (1.0 / stride)))


					p23456[i] = tf.slice(pi, [0,0,0,0],tf.concat([[-1,-1], target_shape], axis=0))
					p23456[i].set_shape([1, pi.shape[1], None,None])

				shape2d = tf.shape(p23456[i])[2:] # h,W
				slice3d = tf.concat([shape2d, [-1]],axis=0)
				slice4d = tf.concat([shape2d, [-1,-1]],axis=0)

				anchors[i] = tf.slice(anchors[i], [0,0,0,0], slice4d)
				self.sliced_anchor_labels.append(tf.slice(gt_anchor_labels[i], [0, 0, 0], slice3d))
				self.sliced_anchor_boxes.append(tf.slice(gt_anchor_boxes[i], [0, 0, 0, 0], slice4d))


	def generate_fpn_proposals(self, multilevel_anchors, multilevel_label_logits,multilevel_box_logits, image_shape2d):
		config = self.config
		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_anchors)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_label_logits)
		all_boxes = []
		all_scores = []
		fpn_nms_topk = config.rpn_train_post_nms_topk if config.is_train else config.rpn_test_post_nms_topk
		for lvl in xrange(num_lvl):
			with tf.name_scope("Lvl%s"%(lvl+2)):
				anchors = multilevel_anchors[lvl]
				pred_boxes_decoded = decode_bbox_target(multilevel_box_logits[lvl], anchors,decode_clip=config.bbox_decode_clip)

				this_fpn_nms_topk = fpn_nms_topk
				if config.obj_v2 and lvl == 0:
					this_fpn_nms_topk = 4000
				proposal_boxes, proposal_scores = generate_rpn_proposals(tf.reshape(pred_boxes_decoded, [-1,4]), tf.reshape(multilevel_label_logits[lvl], [-1]), image_shape2d, config,pre_nms_topk=this_fpn_nms_topk)
				all_boxes.append(proposal_boxes)
				all_scores.append(proposal_scores)


		proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
		proposal_scores = tf.concat(all_scores, axis=0)  # n
		proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
		proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
		proposal_boxes = tf.gather(proposal_boxes, topk_indices)
		return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')

	# based on box sizes
	def fpn_map_rois_to_levels(self, boxes):

		def tf_area(boxes):
			x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
			return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

		sqrtarea = tf.sqrt(tf_area(boxes))
		level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))
		# RoI levels range from 2~5 (not 6)
		level_ids = [ 
			tf.where(level <= 2),
			tf.where(tf.equal(level, 3)),# problems with ==?
			tf.where(tf.equal(level, 4)),
			tf.where(level >= 5)]

		level_ids = [tf.reshape(x, [-1], name='roi_level%s_id'%(i + 2)) for i, x in enumerate(level_ids)]
		num_in_levels = [tf.size(x, name='num_roi_level%s'%(i + 2)) for i, x in enumerate(level_ids)]

		level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
		return level_ids, level_boxes


	# output_shape is the output feature HxW
	def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
		config = self.config
		assert len(features) == 4
		# Reassign rcnn_boxes to levels # based on box area size
		level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
		all_rois = []

		# Crop patches from corresponding levels
		for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
			with tf.name_scope('roi_level%s'%(i + 2)):
				boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])
				all_rois.append(roi_align(featuremap, boxes_on_featuremap, output_shape))

		# this can fail if using TF<=1.8 with MKL build
		all_rois = tf.concat(all_rois, axis=0)  # NCHW
		# Unshuffle to the original order, to match the original samples
		level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
		level_id_invert_perm = tf.invert_permutation(level_id_perm)
		all_rois = tf.gather(all_rois, level_id_invert_perm)
		return all_rois

	def cascade_rcnn_head(self,boxes,stage,p23456):
		config = self.config
		if config.is_train:
			boxes, labels_per_box, fg_inds_wrt_gt = boxes
		reg_weight = config.cascade_bbox_reg[stage]
		reg_weight = tf.constant(reg_weight,dtype=tf.float32)

		pool_feat = self.multilevel_roi_align(p23456[:4],boxes, 7)
		pool_feat = self.scale_gradient_func(pool_feat)

		#box_logits -> [N,1,4]
		# label -> [N, num_class]
		label_logits, box_logits = self.fastrcnn_2fc_head_class_agnostic(pool_feat,config.num_class,boxes=boxes)
		
		refined_boxes = decode_bbox_target(tf.reshape(box_logits, [-1,4]) / reg_weight, boxes)
		refined_boxes = clip_boxes(refined_boxes, tf.shape(self.p_image)[2:])

		# [N], [N,4]
		return label_logits, box_logits, tf.stop_gradient(refined_boxes)

	def match_box_with_gt(self,boxes, iou_threshold):
		config = self.config
		gt_boxes = self.gt_boxes
		if config.is_train:
			with tf.name_scope('match_box_with_gt_%s'%(iou_threshold)):
				iou = pairwise_iou(boxes, gt_boxes)# NxM
				max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
				best_iou_ind = tf.argmax(iou, axis=1)  # N
				labels_per_box = tf.gather(self.gt_labels, best_iou_ind)
				fg_mask = max_iou_per_box >= iou_threshold
				fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
				labels_per_box = tf.stop_gradient(labels_per_box * tf.to_int64(fg_mask))
				return (boxes, labels_per_box, fg_inds_wrt_gt)
		else:
			return boxes

	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		multilevel_anchors = self.multilevel_anchors # a list of numpy anchors, not sliced

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		
		c2345 = resnet_fpn_backbone(image,config.resnet_num_block,resolution_requirement=config.fpn_resolution_requirement,tf_pad_reverse=config.new_tensorpack_model,finer_resolution=config.finer_resolution,freeze=config.freeze)
		

		# include lateral 1x1 conv and final 3x3 conv
		# -> [7,7,256]
		p23456 = fpn_model(c2345,num_channel=config.fpn_num_channel,scope="fpn")

		# [1, H, W, channel]
		self.fpn_feature = tf.image.resize_images(tf.transpose(p23456[3],perm=[0,2,3,1]), (7,7)) # p5 # default bilinear

		if config.no_obj_detect: # pair with extract_feat, so only extract feature
			return True
		# given the numpy anchor for each stride, 
		# slice the anchor box and label against the feature map size on each level. Again?
		self.slice_feature_and_anchors(image_shape2d,p23456,multilevel_anchors)
		# now multilevel_anchors are sliced and tf type
		# added sliced gt anchor labels and boxes
		# so we have each fpn level's anchor boxes, and the ground truth anchor boxes & labels if training

		# given [1,256,FS,FS] feature, each level got len(anchor_ratios) anchor outputs
		rpn_outputs = [self.rpn_head(pi, config.fpn_num_channel, len(config.anchor_ratios), data_format="NCHW",scope="rpn") for pi in p23456]
		multilevel_label_logits = [k[0] for k in rpn_outputs]
		multilevel_box_logits = [k[1] for k in rpn_outputs]

		proposal_boxes, proposal_scores = self.generate_fpn_proposals(multilevel_anchors, multilevel_label_logits, multilevel_box_logits, image_shape2d)

		if config.is_train:
			gt_boxes = self.gt_boxes
			gt_labels = self.gt_labels
			# for training, use gt_box and some proposal box as pos and neg
			# rcnn_sampled_boxes [N_FG+N_NEG,4]
			# fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
			rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(proposal_boxes, gt_boxes,gt_labels,config=config)
		else:
			rcnn_boxes = proposal_boxes


		if config.is_cascade_rcnn:
			if config.is_train:
				@tf.custom_gradient
				def scale_gradient(x):
					return x, lambda dy: dy * (1.0 / config.cascade_num_stage)
				self.scale_gradient_func = scale_gradient
				proposals = (rcnn_boxes, rcnn_labels, fg_inds_wrt_gt)
			else:
				self.scale_gradient_func = tf.identity
				proposals = rcnn_boxes

			# cascade to refine region proposal
			# each step add in groundtruth
			with tf.variable_scope("cascade_rcnn_stage1"):
				# [N,num_class] ,[N,1,4]
				B1_label_logits, B1_box_logits, B1 = self.cascade_rcnn_head(proposals,0,p23456)
			with tf.variable_scope("cascade_rcnn_stage2"):
				B1_proposal = self.match_box_with_gt(B1,config.cascade_ious[1])
				B2_label_logits, B2_box_logits, B2 = self.cascade_rcnn_head(B1_proposal,1,p23456)
			with tf.variable_scope("cascade_rcnn_stage3"):
				B2_proposal = self.match_box_with_gt(B2,config.cascade_ious[2])
				# [N,num_class] ,[N,1,4]
				B3_label_logits, B3_box_logits, B3 = self.cascade_rcnn_head(B2_proposal,2,p23456)

			cascade_box_proposals = [proposals, B1_proposal, B2_proposal]
			cascade_box_logits = [B1_box_logits, B2_box_logits, B3_box_logits]
			cascade_label_logits = [B1_label_logits, B2_label_logits, B3_label_logits]

		else:
			# NxCx7x7 # (?, 256, 7, 7)
			roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],rcnn_boxes, 7)
			# (N,81) ,(N, 80,4)
			if config.obj_v2 or config.obj_v3:
				fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_2fc_head_class_agnostic(roi_feature_fastrcnn,config.num_class,boxes=rcnn_boxes,scope="fastrcnn")
			else:
				fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_2fc_head(roi_feature_fastrcnn,config.num_class,boxes=rcnn_boxes,scope="fastrcnn")
		

		if config.is_train:
			rpn_label_loss, rpn_box_loss = self.multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)

			
			if config.is_cascade_rcnn:
				losses = [rpn_label_loss, rpn_box_loss]


				for i, proposals in enumerate(cascade_box_proposals):
					with tf.name_scope('cascade_loss_stage%s'%(i + 1)):
						this_rcnn_boxes, this_rcnn_labels, this_fg_inds_wrt_gt = cascade_box_proposals[i]
						this_box_logits = cascade_box_logits[i]
						this_label_logits = cascade_label_logits[i]
						reg_weight = config.cascade_bbox_reg[i]

						this_fg_inds_wrt_sample = tf.reshape(tf.where(this_rcnn_labels > 0), [-1])
						this_fg_sampled_boxes = tf.gather(this_rcnn_boxes, this_fg_inds_wrt_sample)
						this_fg_fastrcnn_box_logits = tf.gather(this_box_logits, this_fg_inds_wrt_sample)

						this_matched_gt_boxes = tf.gather(self.gt_boxes, this_fg_inds_wrt_gt)

						this_encoded_boxes = encode_bbox_target(this_matched_gt_boxes, this_fg_sampled_boxes) * tf.constant(reg_weight,dtype=tf.float32) 

						this_label_loss, this_box_loss = self.fastrcnn_losses(this_rcnn_labels, this_label_logits, this_encoded_boxes, this_fg_fastrcnn_box_logits)

						losses.extend([this_label_loss,this_box_loss])

				# for debug
				self.rpn_label_loss = rpn_label_loss
				self.rpn_box_loss = rpn_box_loss
				self.fastrcnn_label_loss = losses[-2]
				self.fastrcnn_box_loss = losses[-1]



			else:#---------------- get fast rcnn loss

				# rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
				fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])

				# for training, maskRCNN only apply on positive box
				# [N_FG, num_class, 14, 14]

				# [N_FG, 4]
				# sampled boxes are at least iou with a gt_boxes
				fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
				fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

				# [N_FG, 4] # each proposal box assigned gt box, may repeat
				matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

				# fastrcnn also need to regress box (just the FG box)
				encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * tf.constant(config.fastrcnn_bbox_reg_weights) #[10,10,5,5]?

				# fastrcnn input is fg and bg proposal box, do classification to num_class(include bg) and then regress on fg boxes
				# [N_FG+N_NEG,4] & [N_FG,4]
				fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_losses(rcnn_labels, fastrcnn_label_logits,encoded_boxes, fg_fastrcnn_box_logits)
				# ---------------------------------------------------------

				# for debug
				self.rpn_label_loss = rpn_label_loss
				self.rpn_box_loss = rpn_box_loss
				self.fastrcnn_label_loss = fastrcnn_label_loss
				self.fastrcnn_box_loss = fastrcnn_box_loss

				losses = [rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss]


			# mask rcnn loss
			if config.add_mask:
				fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])
				fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)

				# NxCx14x14
				# only the fg boxes
				roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],fg_sampled_boxes, 14)

				mask_logits = self.maskrcnn_up4conv_head(fg_feature, config.num_class,scope='maskrcnn')


				# [N_FG, H,W]
				gt_mask = self.gt_mask
				gt_mask_for_fg = tf.gather(gt_mask,fg_inds_wrt_gt)
				# [N_FG, H, W] -> [N_FG, 14, 14]
				target_masks_for_fg = crop_and_resize(
					tf.expand_dims(gt_masks,1), 
					fg_sampled_boxes,
					fg_inds_wrt_gt, 28, pad_border=False) # fg x 1x28x28
				target_masks_for_fg = tf.squeeze(target_masks_for_fg,1)

				mrcnn_loss = self.maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)

				losses+=[mrcnn_loss]

			if config.wd is not None:
				wd = wd_cost('.*/W', config.wd,scope="wd_cost")
				losses.append(wd)

			self.loss = tf.add_n(losses,'total_loss')

			# l2loss
		else:


			# inferencing
			# K -> proposal box
			# [K,num_class]
			#image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits
			if config.is_cascade_rcnn:
				# B3 is already decoded
				decoded_boxes = tf.expand_dims(B3, 1)     # class-agnostic
				decoded_boxes = tf.tile(decoded_boxes, [1, self.num_class-1, 1])

				label_probs = []
				for  i, this_label_logits in enumerate(cascade_label_logits):
					this_label_prob = tf.nn.softmax(this_label_logits)
					label_probs.append(this_label_prob)

				label_probs = tf.multiply(tf.add_n(label_probs), (1.0 / config.cascade_num_stage))


			else:
				
				# get the regressed actual boxes
				# anchor box [K,4] -> [K,num_class - 1, 4] <- box regress logits [K,num_class-1,4]
				anchors = tf.tile(tf.expand_dims(rcnn_boxes,1),[1, config.num_class-1,1])
				if config.obj_v2 or config.obj_v3:
					anchors = tf.expand_dims(rcnn_boxes,1)

				decoded_boxes = decode_bbox_target(fastrcnn_box_logits / tf.constant(config.fastrcnn_bbox_reg_weights,dtype=tf.float32), anchors)

				if config.obj_v2 or config.obj_v3:
					decoded_boxes = tf.tile(decoded_boxes, [1, config.num_class-1, 1])

				label_probs = tf.nn.softmax(fastrcnn_label_logits)

			decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")
				#self.decoded_boxes = decoded_boxes

			# decoded boxes are [K,num_class-1,4]. so from each proposal boxes generate all classses' boxes, with prob, then do nms on these
			# pred_indices: [R,2] , each entry (#proposal[1-K], #catid [0,num_class-1])
			# final_probs [R]
			# here do nms,
			pred_indices, final_probs = self.fastrcnn_predictions(decoded_boxes, label_probs)
			# [R,4]
			final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name="final_boxes")
			# [R] , each is 1-80 catogory
			final_labels = tf.add(pred_indices[:,1],1,name="final_labels")

			if config.add_mask:

				roi_feature_maskrcnn = self.multilevel_roi_align(p23456[:4], final_boxes, 14)

				mask_logits = self.maskrcnn_up4conv_head(roi_feature_maskrcnn, config.num_class,scope='maskrcnn')

				indices = tf.stack([tf.range(tf.size(final_labels)),tf.to_int32(final_labels)-1],axis=1)

				final_mask_logits = tf.gather_nd(mask_logits,indices)
				final_masks = tf.sigmoid(final_mask_logits)
				
				# [R,14,14]
				self.final_masks = final_masks


			# [R,4]
			self.final_boxes = final_boxes
			# [R]
			self.final_labels = final_labels
			self.final_probs = final_probs
			

	# ----some model component
	# feature map -> [1,1024,FS1,FS2] , FS1 = H/16.0, FS2 = W/16.0
	# channle -> 1024
	def rpn_head(self,featuremap, channel, num_anchors, data_format,scope="rpn"):
		with tf.variable_scope(scope):
			# [1, channel, FS1, FS2] # channel = 1024
			# conv0:W -> [3,3,1024,1024]
			h = conv2d(featuremap,channel,kernel=3,activation=tf.nn.relu,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="conv0")
			# h -> [1,1024(channel),FS1,FS2]

			# 1x1 kernel conv to classification on each grid
			# [1, 1024, FS1, FS2] -> # [1, num_anchors, FS1, FS2]
			label_logits = conv2d(h,num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="class")
			# [1, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
			box_logits = conv2d(h,4*num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="box")

			# [1,1024,FS1, FS2] -> [FS1, FS2,1024]
			label_logits = tf.squeeze(tf.transpose(label_logits, [0,2,3,1]),0)

			box_shape = tf.shape(box_logits)
			box_logits = tf.transpose(box_logits,[0,2,3,1]) # [1,FS1, FS2,1024*4]
			# [FS1, FS2,1024,4]
			box_logits = tf.reshape(box_logits,[box_shape[2], box_shape[3],num_anchors,4])

			return label_logits,box_logits

	# feature: [K,C,7,7] # feature for each roi
	def fastrcnn_2fc_head(self,feature,num_class=None,boxes=None,scope="fastrcnn_head"):
		config = self.config
		dim = config.fpn_frcnn_fc_head_dim # 1024
		initializer = tf.variance_scaling_initializer()

		with tf.variable_scope(scope):
			# dense will reshape to [k,C*7*7] first
			if config.add_relation_nn:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
			else:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")


			with tf.variable_scope("outputs"):

				classification = dense(hidden,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
				
			
				if config.new_tensorpack_model:
					box_regression = dense(hidden,num_class*4 ,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					box_regression = tf.reshape(box_regression, (-1, num_class,4))

					box_regression = box_regression[:,1:,:]
					
					box_regression.set_shape([None,num_class-1,4])
				else:
					box_regression = dense(hidden,(num_class -1)*4, W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					
					box_regression = tf.reshape(box_regression, (-1, num_class-1,4))
			
			

		return classification,box_regression

	def fastrcnn_2fc_head_class_agnostic(self,feature,num_class,boxes=None,scope="head"):
		config = self.config
		dim = config.fpn_frcnn_fc_head_dim # 1024
		initializer = tf.variance_scaling_initializer()

		with tf.variable_scope(scope):
			# dense will reshape to [k,C*7*7] first
			if config.add_relation_nn:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
			else:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")


		with tf.variable_scope("outputs"):

			
			classification = dense(hidden,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
			num_class = 1 # just for box
			box_regression = dense(hidden,num_class*4 ,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
			box_regression = tf.reshape(box_regression, (-1, num_class,4))
			
		return classification,box_regression

	
	def maskrcnn_up4conv_head(self,feature,num_class,scope="maskrcnn_head"):
		config = self.config
		num_conv = 4 # C4 model this is 0
		l = feature
		with tf.variable_scope(scope):
			for k in xrange(num_conv):
				l = conv2d(l, config.mrcnn_head_dim, kernel=3, activation=tf.nn.relu, data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='truncated_normal'), scope="fcn%s"%(k))

			l = deconv2d(l, config.mrcnn_head_dim, kernel=2, stride=2, activation=tf.nn.relu, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out", distribution='truncated_normal'), scope="deconv")
			l = conv2d(l,num_class-1, kernel=1, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='normal'), scope="conv")
			return l

	# given all proposal box prediction, based on score thres , get final NMS resulting box
	# [K,num_class-1,4] -> decoded_boxes
	# [K,num_class] label_probs
	# each proposal box has prob and box to all class
	# here using nms for each class, -> [R]
	def fastrcnn_predictions(self,boxes, probs,scope="fastrcnn_predictions"):
		with tf.variable_scope(scope):		
			config = self.config
			assert boxes.shape[1] == config.num_class - 1,(boxes.shape,config.num_class)
			assert probs.shape[1] == config.num_class,(probs.shape[1],config.num_class)
			# transpose to map_fn along each class
			boxes = tf.transpose(boxes,[1,0,2]) # [num_class-1, K,4]
			probs = tf.transpose(probs[:,1:],[1,0]) # [num_class-1, K]

			def f(X):
				prob,box = X # [K], [K,4]
				output_shape = tf.shape(prob)
				# [K]
				ids = tf.reshape(tf.where(prob > config.result_score_thres),[-1])
				prob = tf.gather(prob,ids)
				box = tf.gather(box,ids)
				# NMS
				selection = tf.image.non_max_suppression(box,prob,max_output_size=config.result_per_im,iou_threshold=config.fastrcnn_nms_iou_thres)
				selection = tf.to_int32(tf.gather(ids,selection))
				sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

				mask = tf.sparse_to_dense(sparse_indices=sorted_selection,output_shape=output_shape,sparse_values=True,default_value=False)
				return mask

			# for each catagory get the top K
			# [num_class-1, R]
			masks = tf.map_fn(f, (probs,boxes), dtype=tf.bool, parallel_iterations=10)
			# [R,2], each entry is [cat_id,box_id]
			selected_indices = tf.where(masks)

			probs = tf.boolean_mask(probs,masks)# [num_class-1, K] 

			# topk_indices [num_class-1,result_num]
			topk_probs, topk_indices = tf.nn.top_k(probs, tf.minimum(config.result_per_im,tf.size(probs)),sorted=False)

			# [K,2] <- select [act_num,R] 
			filtered_selection = tf.gather(selected_indices, topk_indices)
			filtered_selection = tf.reverse(filtered_selection, axis=[1],name="filtered")

			# [R,2], [R,]
			return filtered_selection, topk_probs


	# ---- losses
	def maskrcnn_loss(self,mask_logits, fg_labels, fg_target_masks,scope="maskrcnn_loss"):
		with tf.variable_scope(scope):
			# mask_logits: [N_FG, num_cat, 14, 14]
			# fg_labels: [N_FG]
			# fg_target_masks: [N_FG, 14, 14]
			num_fg = tf.size(fg_labels)
			# [N_FG, 2] # these index is used to get the pos cat's logit
			indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1],axis=1)
			# ignore other class's logit
			# [N_FG, 14, 14]
			mask_logits = tf.gather_nd(mask_logits, indices)
			mask_probs = tf.sigmoid(mask_logits)

			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
			loss = tf.reduce_mean(loss, name='maskrcnn_loss')

			return loss


	def multilevel_rpn_losses(self, multilevel_anchors, multilevel_label_logits, multilevel_box_logits, scope="rpn_losses"):
		config = self.config
		sliced_anchor_labels = self.sliced_anchor_labels 
		sliced_anchor_boxes = self.sliced_anchor_boxes

		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_label_logits)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_anchors)

		losses = []
		with tf.variable_scope(scope):
			for lvl in xrange(num_lvl):
				anchors = multilevel_anchors[lvl]
				gt_labels = sliced_anchor_labels[lvl]
				gt_boxes = sliced_anchor_boxes[lvl]

				# get the ground truth T_xywh
				encoded_gt_boxes = encode_bbox_target(gt_boxes, anchors)

				label_loss, box_loss = self.rpn_losses(gt_labels, encoded_gt_boxes, multilevel_label_logits[lvl], multilevel_box_logits[lvl],scope="level%s"%(lvl+2))
				losses.extend([label_loss,box_loss])

			total_label_loss = tf.add_n(losses[::2], name='label_loss')
			total_box_loss = tf.add_n(losses[1::2], name='box_loss')

		return total_label_loss, total_box_loss

			
	def rpn_losses(self, anchor_labels, anchor_boxes, label_logits, box_logits,scope="rpn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
			# label_logits [FS,FS,num_anchors] [7,7,1024]
			# box_logits [FS,FS,num_anchors,4] [7.7,1024,4]
			
			#with tf.device("/cpu:0"):
			valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1)) # 1,0|pos/neg
			pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
			nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
			nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')

			# [K1]

			valid_anchor_labels = tf.boolean_mask(anchor_labels,valid_mask)

			# [K2]
			valid_label_logits = tf.boolean_mask(label_logits, valid_mask)


			placeholder = 0.

			# label loss for all valid anchor box
			if config.focal_loss:
				label_loss = focal_loss(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))
			else:
				label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))

				label_loss = tf.reduce_mean(label_loss,name="label_loss")

				label_loss = tf.reduce_sum(label_loss) * (1. / config.rpn_batch_per_im)

			label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

			# box loss for positive anchor
			pos_anchor_boxes = tf.boolean_mask(anchor_boxes,pos_mask)
			pos_box_logits = tf.boolean_mask(box_logits,pos_mask)

			delta = 1.0/9

			# the smooth l1 loss
			box_loss = tf.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta, reduction=tf.losses.Reduction.SUM) / delta

			#box_loss = tf.div(box_loss, tf.cast(nr_valid, tf.float32),name='box_loss')
			box_loss = box_loss * (1. / config.rpn_batch_per_im)
			box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')


			return label_loss, box_loss

	def fastrcnn_losses(self, labels, label_logits, fg_boxes, fg_box_logits,scope="fastrcnn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# label -> label for roi [N_FG + N_NEG]
			# label_logits [N_FG + N_NEG,num_class]
			# fg_boxes_logits -> [N_FG,num_class-1,4]

			# so the label is int [0-num_class], 0 being background

			if config.focal_loss:
				onehot_label = tf.one_hot(labels,label_logits.get_shape()[-1])

				# here uses sigmoid
				label_loss = focal_loss(logits=label_logits,labels=tf.to_float(onehot_label))
			else:
				label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)

				label_loss = tf.reduce_mean(label_loss, name="label_loss")

			fg_inds = tf.where(labels > 0)[:,0]
			fg_labels = tf.gather(labels, fg_inds) # [N_FG]

			num_fg = tf.size(fg_inds) # N_FG
			if int(fg_box_logits.shape[1]) > 1:
				# [N_FG, 2]
				indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1], axis=1)
				# gather the logits from [N_FG,num_class-1, 4] to [N_FG,4], only the gt class's logit
				fg_box_logits = tf.gather_nd(fg_box_logits, indices)
			else:
				fg_box_logits = tf.reshape(fg_box_logits, [-1, 4]) # class agnostic for cascade rcnn
			box_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

			# /  N_FG + N_NEG ?
			box_loss = tf.truediv(box_loss, tf.to_float(tf.shape(labels)[0]),name='box_loss')

			return label_loss, box_loss


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,batch,is_train=False):

		#{"imgs":[],"gt":[]}
		config = self.config
		
		N = len(batch.data['imgs'])

		assert N == 1 # only 1 image for now

		image = batch.data['imgs'][0]

		feed_dict = {}

		if batch.data.has_key("imgdata"):
			image = batch.data['imgdata'][0]
		else:
			image = cv2.imread(image,cv2.IMREAD_COLOR)
			assert image is not None,image
			image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		# resize image, boxes
		short_edge_size = config.short_edge_size
		if config.scale_jitter and is_train:
			short_edge_size = random.randint(config.short_edge_size_min,config.short_edge_size_max)


		if batch.data.has_key("resized_image"):
			resized_image = batch.data['resized_image'][0]
		else:
			resized_image = resizeImage(image,short_edge_size,config.max_size)
		newh,neww = resized_image.shape[:2]
		#print newh,neww, batch.data['imgs'][0]
		#sys.exit()

		if is_train:
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box
			labels = anno['labels']
			assert len(labels) == len(o_boxes)

			# boxes # (x,y,w,h)
			"""
			boxes = o_boxes[:,[0,2,1,3]] #(x,w,y,h)
			boxes = boxes.reshape((-1,2,2)) #
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x,w
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y,h
			"""

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2


			# random horizontal flip
			# no flip for surveilance video?
			if config.flip_image:
				prob = 0.5
				rand = random.random()
				if rand > prob:
					resized_image = cv2.flip(resized_image,1) # 1 for horizontal
					#boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
					boxes[:,0] = neww - boxes[:,0]
					boxes[:,0,:] = boxes[:,0,::-1]# (x_min will be x_max after flip)
				

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)


			# visualize?
			if config.vis_pre:
				label_names = [config.classId_to_class[i] for i in labels]
				o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
				boxes_x1x2 = np.asarray([box for box in boxes])
				ori_vis = draw_boxes(image,o_boxes_x1x2,labels=label_names)
				new_vis = draw_boxes(resized_image,boxes_x1x2,labels=label_names)
				imgname = os.path.splitext(os.path.basename(batch.data['imgs'][0]))[0]
				cv2.imwrite("%s.ori.jpg"%os.path.join(config.vis_path,imgname),ori_vis)
				cv2.imwrite("%s.prepro.jpg"%os.path.join(config.vis_path,imgname),new_vis)
				print "viz saved in %s"%config.vis_path
				sys.exit()

			# get rpn anchor labels
			# [fs_im,fs_im,num_anchor,4]
			
				
			multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(resized_image, boxes)

			multilevel_anchor_labels = [l for l,b in multilevel_anchor_inputs]
			multilevel_anchor_boxes = [b for l,b in multilevel_anchor_inputs]
			assert len(multilevel_anchor_labels) == len(multilevel_anchor_boxes) == len(self.anchor_labels) == len(self.anchor_boxes), (len(multilevel_anchor_labels), len(multilevel_anchor_boxes), len(self.anchor_labels),len(self.anchor_boxes) )

			for pl_labels,pl_boxes,in_labels,in_boxes in zip(self.anchor_labels,self.anchor_boxes,multilevel_anchor_labels, multilevel_anchor_boxes):

				feed_dict[pl_labels] = in_labels
				feed_dict[pl_boxes] = in_boxes

				

			assert len(boxes) > 0

			feed_dict[self.gt_boxes] = boxes
			feed_dict[self.gt_labels] = labels

		else:
			
			pass

		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

	def get_feed_dict_forward(self,imgdata):
		feed_dict = {}

		feed_dict[self.image] = imgdata

		feed_dict[self.is_train] = False

		return feed_dict

	# anchor related function for training--------------------

	def filter_box_inside(self, im, boxes):
		h, w = im.shape[:2]
		indices = np.where(
			(boxes[:,0] >= 0) &
			(boxes[:,1] >= 0) &
			(boxes[:,2] <= w) &
			(boxes[:,3] <= h)  
		)[0]
		return indices, boxes[indices,:]
	# for training, given image and box, get anchor box labels
	# [fs_im,fs_im,num_anchor,4] # not fs,
	def get_rpn_anchor_input(self,im,boxes):
		

		config = self.config

		boxes = boxes.copy()

		# [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
		all_anchors_np = np.copy(get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size))

		h,w = im.shape[:2]

		# so image may be smaller than the full anchor size
		#featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
		anchorH, anchorW = all_anchors_np.shape[:2]
		featureh, featurew = anchorH, anchorW

		# [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
		#featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
		#print featuremap_anchors.shape #(46,83,15,4)
		#featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))
		featuremap_anchors_flatten = all_anchors_np.reshape((-1,4))

		# num_in < FS_im*FS_im*num_anchors # [num_in,4]
		inside_ind, inside_anchors = self.filter_box_inside(im,featuremap_anchors_flatten) # the anchor box inside the image
		

		# anchor labels is in {1,-1,0}, -1 means ignore
		# N = num_in
		# [N], [N,4] # only the fg anchor has box value
		anchor_labels,anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# fill back to [fs,fs,num_anchor,4]
		# all anchor outside box is ignored (-1)

		featuremap_labels = -np.ones((featureh * featurew*config.num_anchors,),dtype='int32')
		featuremap_labels[inside_ind] = anchor_labels
		featuremap_labels = featuremap_labels.reshape((featureh,featurew,config.num_anchors))

		featuremap_boxes = np.zeros((featureh * featurew*config.num_anchors,4),dtype='float32')
		featuremap_boxes[inside_ind,:] = anchor_boxes
		featuremap_boxes = featuremap_boxes.reshape((featureh,featurew,config.num_anchors,4))

		return featuremap_labels,featuremap_boxes

	def get_multilevel_rpn_anchor_input(self,im,boxes):

		config = self.config

		boxes = boxes.copy()

		anchors_per_level = self.get_all_anchors_fpn() # get anchor for each (anchor_stride,anchor_size) pair
		flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
		all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
		# some image may not be resized to max size, could be shorter edge size
		inside_ind, inside_anchors = self.filter_box_inside(im,all_anchors_flatten)
		# given all these anchors, given the ground truth box, and their iou to each anchor, get the label to be 1 or 0.
		anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# map back to all_anchors, then split to each level
		num_all_anchors = all_anchors_flatten.shape[0]
		all_labels = -np.ones((num_all_anchors, ), dtype='int32')
		all_labels[inside_ind] = anchor_labels
		all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
		all_boxes[inside_ind] = anchor_gt_boxes

		start = 0
		multilevel_inputs = []

		# put back to list for each level

		for level_anchor in anchors_per_level:
			assert level_anchor.shape[2] == len(config.anchor_ratios)
			anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
			num_anchor_this_level = np.prod(anchor_shape)
			end = start + num_anchor_this_level
			multilevel_inputs.append(
				(all_labels[start: end].reshape(anchor_shape),
				all_boxes[start:end, :].reshape(anchor_shape + (4,))))
			start = end

		assert end == num_all_anchors, ("num all anchors:%s, end:%s"%(num_all_anchors,end))
		return multilevel_inputs




	def get_anchor_labels(self,anchors,gt_boxes):
		config = self.config

		# return max_num of index for labels equal val
		def filter_box_label(labels, val, max_num):
			cur_inds = np.where(labels == val)[0]
			if len(cur_inds) > max_num:
				disable_inds = np.random.choice(cur_inds,size=(len(cur_inds) - max_num),replace=False)
				labels[disable_inds] = -1
				cur_inds = np.where(labels == val)[0]
			return cur_inds

		NA,NB = len(anchors),len(gt_boxes)
		assert NB > 0

		#bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
		#box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
		box_ious = np_iou(anchors, gt_boxes)

		#print box_ious.shape #(37607,7)

		#NA, each anchors max iou to any gt box, and the max gt box's index [0,NB-1]
		iou_argmax_per_anchor = box_ious.argmax(axis=1)
		iou_max_per_anchor = box_ious.max(axis=1)

		# 1 x NB, each gt box's max iou to any anchor boxes
		#iou_max_per_gt = box_ious.max(axis=1,keepdims=True) 
		#print iou_max_per_gt # all zero?
		iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

		# NA x 1? True for anchors that cover all the gt boxes
		anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

		anchor_labels = -np.ones((NA,),dtype='int32')

		anchor_labels[anchors_with_max_iou_per_gt] = 1
		anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
		anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

		# cap the number of fg anchor and bg anchor
		target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

		# set the label==1 to -1 if the number exceeds
		fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

		#assert len(fg_inds) > 0
		old_num_bg = np.sum(anchor_labels == 0)
		if old_num_bg == 0:
			raise Exception("No valid background for RPN!")

		# the rest of 256 is negative
		target_num_bg = config.rpn_batch_per_im - len(fg_inds)

		# set some label to -1 if exceeds
		filter_box_label(anchor_labels,0,target_num_bg)

		# only the fg anchor_boxes are filled with the corresponding gt_box
		anchor_boxes = np.zeros((NA,4),dtype='float32')
		anchor_boxes[fg_inds,:] = gt_boxes[iou_argmax_per_anchor[fg_inds],:]
		return anchor_labels, anchor_boxes

class Mask_RCNN_FPN_Act():
	def __init__(self,config,gpuid=0):
		self.gpuid = gpuid
		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model

		self.config = config

		self.num_class = config.num_class
		self.num_act_class = config.num_act_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		# for training
		self.anchor_labels = []
		self.anchor_boxes = []
		num_anchors = len(config.anchor_ratios)
		for k in xrange(len(config.anchor_strides)):
			self.anchor_labels.append(tf.placeholder(tf.int32,[None, None, num_anchors],name="anchor_labels_lvl%s"%(k+2)))
			self.anchor_boxes.append(tf.placeholder(tf.float32,[None, None, num_anchors,4],name="anchor_boxes_lvl%s"%(k+2)))

		self.gt_boxes = tf.placeholder(tf.float32,[None, 4],name="gt_boxes")
		self.gt_labels = tf.placeholder(tf.int64,[None,],name="gt_labels")

		self.act_boxes = tf.placeholder(tf.float32,[None, 4],name="act_boxes")
		self.act_labels = tf.placeholder(tf.int64,[None,],name="act_labels")

		self.gt_mask = tf.placeholder(tf.uint8,[None, None, None],name="gt_masks") # H,W,v -> {0,1}

		# the following will be added in the build_forward and loss
		self.logits = None
		self.yp = None
		self.loss = None

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		self.multilevel_anchors = []
		with tf.name_scope("fpn_anchors"):#,tf.device("/cpu:0"):
			#fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor box coordinates for a given max_size image,
			# so for 1920 x 1920 image, 1290/16 = 120, so (120,120,NA,4) box, NA is scale*ratio boxes
			self.multilevel_anchors = self.get_all_anchors_fpn()


		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)

			
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			p_image = p_image*(1.0/255)


			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image

	def get_all_anchors_fpn(self):
		config = self.config
		anchors = []
		assert len(config.anchor_strides) == len(config.anchor_sizes)
		for stride, size in zip(config.anchor_strides, config.anchor_sizes):
			anchors_np = get_all_anchors(stride=stride,sizes=[size],ratios=config.anchor_ratios,max_size=config.max_size) 
		
			anchors.append(anchors_np)
		return anchors
		

	def slice_feature_and_anchors(self,image_shape2d,p23456,anchors):
		# anchors is the numpy anchors for different levels
		config = self.config
		# the anchor labels and boxes are grouped into 
		gt_anchor_labels = self.anchor_labels
		gt_anchor_boxes = self.anchor_boxes
		self.sliced_anchor_labels = []
		self.sliced_anchor_boxes = []
		for i,stride in enumerate(config.anchor_strides):
			with tf.name_scope("FPN_slice_lvl%s"%(i)):
				if i<3:
					# Images are padded for p5, which are too large for p2-p4.
					pi = p23456[i]
					target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * (1.0 / stride)))


					p23456[i] = tf.slice(pi, [0,0,0,0],tf.concat([[-1,-1], target_shape], axis=0))
					p23456[i].set_shape([1, pi.shape[1], None,None])

				shape2d = tf.shape(p23456[i])[2:] # h,W
				slice3d = tf.concat([shape2d, [-1]],axis=0)
				slice4d = tf.concat([shape2d, [-1,-1]],axis=0)

				anchors[i] = tf.slice(anchors[i], [0,0,0,0], slice4d)
				self.sliced_anchor_labels.append(tf.slice(gt_anchor_labels[i], [0, 0, 0], slice3d))
				self.sliced_anchor_boxes.append(tf.slice(gt_anchor_boxes[i], [0, 0, 0, 0], slice4d))


	def generate_fpn_proposals(self, multilevel_anchors, multilevel_label_logits,multilevel_box_logits, image_shape2d):
		config = self.config
		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_anchors)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_label_logits)
		all_boxes = []
		all_scores = []
		fpn_nms_topk = config.rpn_train_post_nms_topk if config.is_train else config.rpn_test_post_nms_topk
		for lvl in xrange(num_lvl):
			with tf.name_scope("Lvl%s"%(lvl+2)):
				anchors = multilevel_anchors[lvl]
				pred_boxes_decoded = decode_bbox_target(multilevel_box_logits[lvl], anchors,decode_clip=config.bbox_decode_clip)

				proposal_boxes, proposal_scores = generate_rpn_proposals(tf.reshape(pred_boxes_decoded, [-1,4]), tf.reshape(multilevel_label_logits[lvl], [-1]), image_shape2d, config,pre_nms_topk=fpn_nms_topk)
				all_boxes.append(proposal_boxes)
				all_scores.append(proposal_scores)


		proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
		proposal_scores = tf.concat(all_scores, axis=0)  # n
		proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
		proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
		proposal_boxes = tf.gather(proposal_boxes, topk_indices)
		return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')

	# based on box sizes
	def fpn_map_rois_to_levels(self, boxes):

		def tf_area(boxes):
			x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
			return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

		sqrtarea = tf.sqrt(tf_area(boxes))
		level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))
		# RoI levels range from 2~5 (not 6)
		level_ids = [ 
			tf.where(level <= 2),
			tf.where(tf.equal(level, 3)),
			tf.where(tf.equal(level, 4)),
			tf.where(level >= 5)]

		level_ids = [tf.reshape(x, [-1], name='roi_level%s_id'%(i + 2)) for i, x in enumerate(level_ids)]
		num_in_levels = [tf.size(x, name='num_roi_level%s'%(i + 2)) for i, x in enumerate(level_ids)]

		level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
		return level_ids, level_boxes


	# output_shape is the output feature HxW
	def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
		config = self.config
		assert len(features) == 4
		# Reassign rcnn_boxes to levels # based on box area size
		level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
		all_rois = []

		# Crop patches from corresponding levels
		for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
			with tf.name_scope('roi_level%s'%(i + 2)):
				boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])
				all_rois.append(roi_align(featuremap, boxes_on_featuremap, output_shape))

		# this can fail if using TF<=1.8 with MKL build
		all_rois = tf.concat(all_rois, axis=0)  # NCHW
		# Unshuffle to the original order, to match the original samples
		level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
		level_id_invert_perm = tf.invert_permutation(level_id_perm)
		all_rois = tf.gather(all_rois, level_id_invert_perm)
		return all_rois


	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		multilevel_anchors = self.multilevel_anchors # a list of numpy anchors, not sliced

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		
		c2345 = resnet_fpn_backbone(image,config.resnet_num_block,resolution_requirement=config.fpn_resolution_requirement,tf_pad_reverse=config.new_tensorpack_model,freeze=config.freeze)


		# include lateral 1x1 conv and final 3x3 conv
		p23456 = fpn_model(c2345,num_channel=config.fpn_num_channel,scope="fpn")

		if config.fix_obj_model:
			p2,p3,p4,p5,p6 = p23456
			p23456 = [tf.stop_gradient(p2),tf.stop_gradient(p3),tf.stop_gradient(p4),tf.stop_gradient(p5),tf.stop_gradient(p6)]

		# [1, H, W, channel]
		self.fpn_feature = tf.image.resize_images(tf.transpose(p23456[3],perm=[0,2,3,1]), (7,7)) # p5 # default bilinear


		# given the numpy anchor for each stride, 
		# slice the anchor box and label against the feature map size on each level. Again?
		self.slice_feature_and_anchors(image_shape2d,p23456,multilevel_anchors)
		# now multilevel_anchors are sliced and tf type
		# added sliced gt anchor labels and boxes
		# so we have each fpn level's anchor boxes, and the ground truth anchor boxes & labels if training

		# given [1,256,FS,FS] feature, each level got len(anchor_ratios) anchor outputs
		rpn_outputs = [self.rpn_head(pi, config.fpn_num_channel, len(config.anchor_ratios), data_format="NCHW",freeze=config.fix_obj_model,scope="rpn") for pi in p23456]

		multilevel_label_logits = [k[0] for k in rpn_outputs]
		multilevel_box_logits = [k[1] for k in rpn_outputs]

		proposal_boxes, proposal_scores = self.generate_fpn_proposals(multilevel_anchors, multilevel_label_logits, multilevel_box_logits, image_shape2d)

		if config.is_train:
			gt_boxes = self.gt_boxes
			gt_labels = self.gt_labels
			# for training, use gt_box and some proposal box as pos and neg
			# rcnn_sampled_boxes [N_FG+N_NEG,4]
			# fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
			rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(proposal_boxes, gt_boxes,gt_labels,config=config)
			# separately sample act boxes
			act_rcnn_boxes, act_rcnn_labels, act_fg_inds_wrt_gt = sample_fast_rcnn_targets(proposal_boxes, self.act_boxes,self.act_labels,config=config,fg_ratio=config.fastrcnn_act_fg_ratio)
		else:
			rcnn_boxes = proposal_boxes
			act_rcnn_boxes = proposal_boxes

		# fast rcnn for object
		# NxCx7x7 # (?, 256, 7, 7)
		roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],rcnn_boxes, 7)
		# (N,81) ,(N, 80,4)
		fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_2fc_head(roi_feature_fastrcnn,config.num_class,boxes=rcnn_boxes,freeze=config.fix_obj_model,scope="fastrcnn")


		# change from 7 to 14 to get more feature since bigger boxes?
		#act_roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],act_rcnn_boxes, 14)
		act_roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],act_rcnn_boxes, 7)
		# (N,81) ,(N, 80,4)
		act_fastrcnn_label_logits, act_fastrcnn_box_logits = self.fastrcnn_2fc_head(act_roi_feature_fastrcnn,config.num_act_class,boxes=act_rcnn_boxes,relation=config.act_relation_nn,scope="fastrcnn_activity")


		if config.is_train:
			rpn_label_loss, rpn_box_loss = self.multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)

			
			losses = [rpn_label_loss, rpn_box_loss]
			# object fastrcnn loss
			#---------------- get fast rcnn loss

			# rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
			fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])


			# for training, maskRCNN only apply on positive box
			# [N_FG, num_class, 14, 14]

			# [N_FG, 4]
			# sampled boxes are at least iou with a gt_boxes
			fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
			fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

			# [N_FG, 4] # each proposal box assigned gt box, may repeat
			matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

			# fastrcnn also need to regress box (just the FG box)
			encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * tf.constant(config.fastrcnn_bbox_reg_weights) #[10,10,5,5]?

			# fastrcnn input is fg and bg proposal box, do classification to num_class(include bg) and then regress on fg boxes
			# [N_FG+N_NEG,4] & [N_FG,4]
			fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_losses(rcnn_labels, fastrcnn_label_logits,encoded_boxes, fg_fastrcnn_box_logits)
			# ---------------------------------------------------------

			# for debug
			self.rpn_label_loss = rpn_label_loss
			self.rpn_box_loss = rpn_box_loss
			self.fastrcnn_label_loss = fastrcnn_label_loss
			self.fastrcnn_box_loss = fastrcnn_box_loss

			losses.extend([fastrcnn_label_loss, fastrcnn_box_loss])

			self.act_losses = []

			
			# rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
			act_fg_inds_wrt_sample = tf.reshape(tf.where(act_rcnn_labels > 0), [-1])


			# for training, maskRCNN only apply on positive box
			# [N_FG, num_class, 14, 14]

			# [N_FG, 4]
			# sampled boxes are at least iou with a gt_boxes
			act_fg_sampled_boxes = tf.gather(act_rcnn_boxes, act_fg_inds_wrt_sample)
			act_fg_fastrcnn_box_logits = tf.gather(act_fastrcnn_box_logits, act_fg_inds_wrt_sample)

			# [N_FG, 4] # each proposal box assigned gt box, may repeat
			act_matched_gt_boxes = tf.gather(self.act_boxes, act_fg_inds_wrt_gt)

			# fastrcnn also need to regress box (just the FG box)
			act_encoded_boxes = encode_bbox_target(act_matched_gt_boxes, act_fg_sampled_boxes) * tf.constant(config.fastrcnn_bbox_reg_weights) #[10,10,5,5]?

			# fastrcnn input is fg and bg proposal box, do classification to num_class(include bg) and then regress on fg boxes
			# [N_FG+N_NEG,4] & [N_FG,4]
			act_fastrcnn_label_loss, act_fastrcnn_box_loss = self.fastrcnn_losses(act_rcnn_labels, act_fastrcnn_label_logits,act_encoded_boxes, act_fg_fastrcnn_box_logits)

			act_fastrcnn_label_loss = act_fastrcnn_label_loss*tf.constant(config.act_loss_weight)
			act_fastrcnn_box_loss = act_fastrcnn_box_loss*tf.constant(config.act_loss_weight)

			self.act_losses.extend([act_fastrcnn_label_loss, act_fastrcnn_box_loss])

			losses.extend([act_fastrcnn_label_loss, act_fastrcnn_box_loss])

			if config.wd is not None:
				wd = wd_cost('.*/W', config.wd,scope="wd_cost")
				losses.append(wd)

			self.loss = tf.add_n(losses,'total_loss')

			# l2loss
		else:

			# get the regressed actual boxes
			# anchor box [K,4] -> [K,num_class - 1, 4] <- box regress logits [K,num_class-1,4]
			anchors = tf.tile(tf.expand_dims(rcnn_boxes,1),[1, config.num_class-1,1])
			decoded_boxes = decode_bbox_target(fastrcnn_box_logits / tf.constant(config.fastrcnn_bbox_reg_weights,dtype=tf.float32), anchors)
			decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")

			label_probs = tf.nn.softmax(fastrcnn_label_logits)

			pred_indices, final_probs = self.fastrcnn_predictions(decoded_boxes, label_probs)
			# [R,4]
			final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name="final_boxes")
			# [R] , each is 1-80 catogory
			final_labels = tf.add(pred_indices[:,1],1,name="final_labels")


			
			act_anchors = tf.tile(tf.expand_dims(act_rcnn_boxes,1),[1, config.num_act_class-1,1])
			act_decoded_boxes = decode_bbox_target(act_fastrcnn_box_logits / tf.constant(config.fastrcnn_bbox_reg_weights,dtype=tf.float32), act_anchors)
			act_decoded_boxes = clip_boxes(act_decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")

			act_label_probs = tf.nn.softmax(act_fastrcnn_label_logits)
			#--------------------------------


			act_pred_indices, act_final_probs = self.fastrcnn_predictions(act_decoded_boxes, act_label_probs,num_class=config.num_act_class)
			# [R,4]
			act_final_boxes = tf.gather_nd(act_decoded_boxes, act_pred_indices, name="final_boxes")
			# [R] , each is 1-80 catogory
			act_final_labels = tf.add(act_pred_indices[:,1],1,name="act_final_labels")


			# [R,4]
			self.final_boxes = final_boxes
			# [R]
			self.final_labels = final_labels
			self.final_probs = final_probs
			
			self.act_final_boxes = act_final_boxes
			# [R]
			self.act_final_labels = act_final_labels
			self.act_final_probs = act_final_probs
			

	# ----some model component
	# feature map -> [1,1024,FS1,FS2] , FS1 = H/16.0, FS2 = W/16.0
	# channle -> 1024
	def rpn_head(self,featuremap, channel, num_anchors, data_format,freeze=False, scope="rpn"):
		with tf.variable_scope(scope):
			# [1, channel, FS1, FS2] # channel = 1024
			# conv0:W -> [3,3,1024,1024]
			h = conv2d(featuremap,channel,kernel=3,activation=tf.nn.relu,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="conv0")
			# h -> [1,1024(channel),FS1,FS2]

			# 1x1 kernel conv to classification on each grid
			# [1, 1024, FS1, FS2] -> # [1, num_anchors, FS1, FS2]
			label_logits = conv2d(h,num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="class")
			# [1, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
			box_logits = conv2d(h,4*num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="box")

			# [1,1024,FS1, FS2] -> [FS1, FS2,1024]
			label_logits = tf.squeeze(tf.transpose(label_logits, [0,2,3,1]),0)

			box_shape = tf.shape(box_logits)
			box_logits = tf.transpose(box_logits,[0,2,3,1]) # [1,FS1, FS2,1024*4]
			# [FS1, FS2,1024,4]
			box_logits = tf.reshape(box_logits,[box_shape[2], box_shape[3],num_anchors,4])

			if freeze:
				box_logits = tf.stop_gradient(box_logits)
				label_logits = tf.stop_gradient(label_logits)

			return label_logits,box_logits

	# feature: [K,C,7,7] # feature for each roi
	def fastrcnn_2fc_head(self,feature,num_class=None,boxes=None,relation=False, freeze=False, scope="fastrcnn_head"):
		config = self.config
		dim = config.fpn_frcnn_fc_head_dim # 1024
		initializer = tf.variance_scaling_initializer()

		with tf.variable_scope(scope):
			# dense will reshape to [k,C*7*7] first
			if config.add_relation_nn or relation:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
			else:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")


			with tf.variable_scope("outputs"):

				classification = dense(hidden,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
			
				if config.new_tensorpack_model:
					box_regression = dense(hidden,num_class*4 ,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					box_regression = tf.reshape(box_regression, (-1, num_class,4))

					box_regression = box_regression[:,1:,:]
					
					box_regression.set_shape([None,num_class-1,4])
				else:
					box_regression = dense(hidden,(num_class -1)*4, W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					
					box_regression = tf.reshape(box_regression, (-1, num_class-1,4))
			
			
		if freeze:
			classification = tf.stop_gradient(classification)
			box_regression = tf.stop_gradient(box_regression)

		return classification,box_regression

	def fastrcnn_2fc_head_class_agnostic(self,feature,num_class,boxes=None):
		config = self.config
		dim = config.fpn_frcnn_fc_head_dim # 1024
		initializer = tf.variance_scaling_initializer()

		with tf.variable_scope("head"):
			# dense will reshape to [k,C*7*7] first
			if config.add_relation_nn:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
			else:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")


		with tf.variable_scope("outputs"):

			
			classification = dense(hidden,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
			num_class = 1 # just for box
			box_regression = dense(hidden,num_class*4 ,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
			box_regression = tf.reshape(box_regression, (-1, num_class,4))
			
		return classification,box_regression

	
	def maskrcnn_up4conv_head(self,feature,num_class,scope="maskrcnn_head"):
		config = self.config
		num_conv = 4 # C4 model this is 0
		l = feature
		with tf.variable_scope(scope):
			for k in xrange(num_conv):
				l = conv2d(l, config.mrcnn_head_dim, kernel=3, activation=tf.nn.relu, data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='truncated_normal'), scope="fcn%s"%(k))

			l = deconv2d(l, config.mrcnn_head_dim, kernel=2, stride=2, activation=tf.nn.relu, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out", distribution='truncated_normal'), scope="deconv")
			l = conv2d(l,num_class-1, kernel=1, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='normal'), scope="conv")
			return l

	# given all proposal box prediction, based on score thres , get final NMS resulting box
	# [K,num_class-1,4] -> decoded_boxes
	# [K,num_class] label_probs
	# each proposal box has prob and box to all class
	# here using nms for each class, -> [R]
	def fastrcnn_predictions(self,boxes, probs,num_class=None, scope="fastrcnn_predictions"):
		with tf.variable_scope(scope):		
			config = self.config
			if num_class is None:
				num_class = config.num_class
			assert boxes.shape[1] == num_class - 1,(boxes.shape,num_class)
			assert probs.shape[1] == num_class,(probs.shape[1],num_class)
			# transpose to map_fn along each class
			boxes = tf.transpose(boxes,[1,0,2]) # [num_class-1, K,4]
			probs = tf.transpose(probs[:,1:],[1,0]) # [num_class-1, K]

			def f(X):
				prob,box = X # [K], [K,4]
				output_shape = tf.shape(prob)
				# [K]
				ids = tf.reshape(tf.where(prob > config.result_score_thres),[-1])
				prob = tf.gather(prob,ids)
				box = tf.gather(box,ids)
				# NMS
				selection = tf.image.non_max_suppression(box,prob,max_output_size=config.result_per_im,iou_threshold=config.fastrcnn_nms_iou_thres)
				selection = tf.to_int32(tf.gather(ids,selection))
				sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

				mask = tf.sparse_to_dense(sparse_indices=sorted_selection,output_shape=output_shape,sparse_values=True,default_value=False)
				return mask

			# for each catagory get the top K
			# [num_class-1, R]
			masks = tf.map_fn(f, (probs,boxes), dtype=tf.bool, parallel_iterations=10)
			# [R,2], each entry is [cat_id,box_id]
			selected_indices = tf.where(masks)

			probs = tf.boolean_mask(probs,masks)# [num_class-1, K] 

			# topk_indices [num_class-1,result_num]
			topk_probs, topk_indices = tf.nn.top_k(probs, tf.minimum(config.result_per_im,tf.size(probs)),sorted=False)

			# [K,2] <- select [act_num,R] 
			filtered_selection = tf.gather(selected_indices, topk_indices)
			filtered_selection = tf.reverse(filtered_selection, axis=[1],name="filtered")

			# [R,2], [R,]
			return filtered_selection, topk_probs


	# ---- losses
	def maskrcnn_loss(self,mask_logits, fg_labels, fg_target_masks,scope="maskrcnn_loss"):
		with tf.variable_scope(scope):
			# mask_logits: [N_FG, num_cat, 14, 14]
			# fg_labels: [N_FG]
			# fg_target_masks: [N_FG, 14, 14]
			num_fg = tf.size(fg_labels)
			# [N_FG, 2] # these index is used to get the pos cat's logit
			indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1],axis=1)
			# ignore other class's logit
			# [N_FG, 14, 14]
			mask_logits = tf.gather_nd(mask_logits, indices)
			mask_probs = tf.sigmoid(mask_logits)

			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
			loss = tf.reduce_mean(loss, name='maskrcnn_loss')

			return loss


	def multilevel_rpn_losses(self, multilevel_anchors, multilevel_label_logits, multilevel_box_logits, scope="rpn_losses"):
		config = self.config
		sliced_anchor_labels = self.sliced_anchor_labels 
		sliced_anchor_boxes = self.sliced_anchor_boxes

		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_label_logits)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_anchors)

		losses = []
		with tf.variable_scope(scope):
			for lvl in xrange(num_lvl):
				anchors = multilevel_anchors[lvl]
				gt_labels = sliced_anchor_labels[lvl]
				gt_boxes = sliced_anchor_boxes[lvl]

				# get the ground truth T_xywh
				encoded_gt_boxes = encode_bbox_target(gt_boxes, anchors)

				label_loss, box_loss = self.rpn_losses(gt_labels, encoded_gt_boxes, multilevel_label_logits[lvl], multilevel_box_logits[lvl],scope="level%s"%(lvl+2))
				losses.extend([label_loss,box_loss])

			total_label_loss = tf.add_n(losses[::2], name='label_loss')
			total_box_loss = tf.add_n(losses[1::2], name='box_loss')

		return total_label_loss, total_box_loss

			
	def rpn_losses(self, anchor_labels, anchor_boxes, label_logits, box_logits,scope="rpn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
			# label_logits [FS,FS,num_anchors] [7,7,1024]
			# box_logits [FS,FS,num_anchors,4] [7.7,1024,4]
			
			#with tf.device("/cpu:0"):
			valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1)) # 1,0|pos/neg
			pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
			nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
			nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')

			# [K1]

			valid_anchor_labels = tf.boolean_mask(anchor_labels,valid_mask)

			# [K2]
			valid_label_logits = tf.boolean_mask(label_logits, valid_mask)


			placeholder = 0.

			# label loss for all valid anchor box
			if config.focal_loss:
				label_loss = focal_loss(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))
			else:
				label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))

				label_loss = tf.reduce_mean(label_loss,name="label_loss")

				label_loss = tf.reduce_sum(label_loss) * (1. / config.rpn_batch_per_im)

			label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

			# box loss for positive anchor
			pos_anchor_boxes = tf.boolean_mask(anchor_boxes,pos_mask)
			pos_box_logits = tf.boolean_mask(box_logits,pos_mask)

			delta = 1.0/9

			# the smooth l1 loss
			box_loss = tf.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta, reduction=tf.losses.Reduction.SUM) / delta

			#box_loss = tf.div(box_loss, tf.cast(nr_valid, tf.float32),name='box_loss')
			box_loss = box_loss * (1. / config.rpn_batch_per_im)
			box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')


			return label_loss, box_loss

	def fastrcnn_losses(self, labels, label_logits, fg_boxes, fg_box_logits,scope="fastrcnn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# label -> label for roi [N_FG + N_NEG]
			# label_logits [N_FG + N_NEG,num_class]
			# fg_boxes_logits -> [N_FG,num_class-1,4], [N_FG,1,4] for cascade rcnn

			# so the label is int [0-num_class], 0 being background

			if config.focal_loss:
				onehot_label = tf.one_hot(labels,label_logits.get_shape()[-1])

				# here uses sigmoid
				label_loss = focal_loss(logits=label_logits,labels=tf.to_float(onehot_label))
			else:
				label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)

				label_loss = tf.reduce_mean(label_loss, name="label_loss")

			fg_inds = tf.where(labels > 0)[:,0]
			fg_labels = tf.gather(labels, fg_inds) # [N_FG]

			num_fg = tf.size(fg_inds) # N_FG
			if int(fg_box_logits.shape[1]) > 1:
				# [N_FG, 2]
				indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1], axis=1)
				# gather the logits from [N_FG,num_class-1, 4] to [N_FG,4], only the gt class's logit
				fg_box_logits = tf.gather_nd(fg_box_logits, indices)
			else:
				fg_box_logits = tf.reshape(fg_box_logits, [-1, 4]) # class agnostic for cascade rcnn
			box_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

			# /  N_FG + N_NEG ?
			box_loss = tf.truediv(box_loss, tf.to_float(tf.shape(labels)[0]),name='box_loss')

			return label_loss, box_loss


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,batch,is_train=False):

		#{"imgs":[],"gt":[]}
		config = self.config
		
		N = len(batch.data['imgs'])

		assert N == 1 # only 1 image for now

		image = batch.data['imgs'][0]

		feed_dict = {}

		if batch.data.has_key("imgdata"):
			image = batch.data['imgdata'][0]
		else:
			image = cv2.imread(image,cv2.IMREAD_COLOR)
			assert image is not None,image
			image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		# resize image, boxes
		short_edge_size = config.short_edge_size
		if config.scale_jitter and is_train:
			short_edge_size = random.randint(config.short_edge_size_min,config.short_edge_size_max)

		if batch.data.has_key("resized_image"):
			resized_image = batch.data['resized_image'][0]
		else:
			resized_image = resizeImage(image,short_edge_size,config.max_size)

		newh,neww = resized_image.shape[:2]

		#print newh,neww,batch.data['imgs'][0]
		#sys.exit()

		if is_train:
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box
			labels = anno['labels']
			act_boxes = anno['actboxes']
			act_labels = anno['actlabels']
			
			assert len(act_labels) == len(act_boxes)
			assert len(labels) == len(o_boxes)

			# boxes # (x,y,w,h)
			"""
			boxes = o_boxes[:,[0,2,1,3]] #(x,w,y,h)
			boxes = boxes.reshape((-1,2,2)) #
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x,w
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y,h
			"""

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2

			# act box cannot be empty
			aboxes = act_boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
			aboxes = aboxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			aboxes[:,0] = aboxes[:,0] * (neww*1.0/w) # x1,x2
			aboxes[:,1] = aboxes[:,1] * (newh*1.0/h) # y1,y2


			# random horizontal flip
			# no flip for surveilance video?
			if config.flip_image:
				prob = 0.5
				rand = random.random()
				if rand > prob:
					resized_image = cv2.flip(resized_image,1) # 1 for horizontal
					#boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
					boxes[:,0] = neww - boxes[:,0]
					boxes[:,0,:] = boxes[:,0,::-1]# (x_min will be x_max after flip)

					aboxes[:,0] = neww - aboxes[:,0]
					aboxes[:,0,:] = aboxes[:,0,::-1]# (x_min will be x_max after flip)
				

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)

			aboxes = aboxes.reshape((-1,4))
			aboxes = aboxes[:,[0,2,1,3]] #(x1,y1,x2,y2)


			# visualize?
			if config.vis_pre:
				label_names = [config.classId_to_class[i] for i in labels]
				o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
				boxes_x1x2 = np.asarray([box for box in boxes])
				ori_vis = draw_boxes(image,o_boxes_x1x2,labels=label_names)
				new_vis = draw_boxes(resized_image,boxes_x1x2,labels=label_names)
				imgname = os.path.splitext(os.path.basename(batch.data['imgs'][0]))[0]
				cv2.imwrite("%s.ori.jpg"%os.path.join(config.vis_path,imgname),ori_vis)
				cv2.imwrite("%s.prepro.jpg"%os.path.join(config.vis_path,imgname),new_vis)
				print "viz saved in %s"%config.vis_path
				sys.exit()

			# get rpn anchor labels
			# [fs_im,fs_im,num_anchor,4]
			
			#multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(resized_image, boxes)
			# RPN target is both the obj boxes and activity boxes?
			multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(resized_image, np.concatenate([boxes,aboxes],axis=0))

			multilevel_anchor_labels = [l for l,b in multilevel_anchor_inputs]
			multilevel_anchor_boxes = [b for l,b in multilevel_anchor_inputs]
			assert len(multilevel_anchor_labels) == len(multilevel_anchor_boxes) == len(self.anchor_labels) == len(self.anchor_boxes), (len(multilevel_anchor_labels), len(multilevel_anchor_boxes), len(self.anchor_labels),len(self.anchor_boxes) )

			for pl_labels,pl_boxes,in_labels,in_boxes in zip(self.anchor_labels,self.anchor_boxes,multilevel_anchor_labels, multilevel_anchor_boxes):

				feed_dict[pl_labels] = in_labels
				feed_dict[pl_boxes] = in_boxes

				

			assert len(boxes) > 0

			feed_dict[self.gt_boxes] = boxes
			feed_dict[self.gt_labels] = labels

			feed_dict[self.act_boxes] = aboxes
			feed_dict[self.act_labels] = act_labels

		else:
			
			pass

		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

	def get_feed_dict_forward(self,imgdata):
		feed_dict = {}

		feed_dict[self.image] = imgdata

		feed_dict[self.is_train] = False

		return feed_dict

	# anchor related function for training--------------------

	def filter_box_inside(self, im, boxes):
		h, w = im.shape[:2]
		indices = np.where(
			(boxes[:,0] >= 0) &
			(boxes[:,1] >= 0) &
			(boxes[:,2] <= w) &
			(boxes[:,3] <= h)  
		)[0]
		return indices, boxes[indices,:]
	# for training, given image and box, get anchor box labels
	# [fs_im,fs_im,num_anchor,4] # not fs,
	def get_rpn_anchor_input(self,im,boxes):
		

		config = self.config

		boxes = boxes.copy()

		# [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
		all_anchors_np = np.copy(get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size))

		h,w = im.shape[:2]

		# so image may be smaller than the full anchor size
		#featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
		anchorH, anchorW = all_anchors_np.shape[:2]
		featureh, featurew = anchorH, anchorW

		# [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
		#featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
		#print featuremap_anchors.shape #(46,83,15,4)
		#featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))
		featuremap_anchors_flatten = all_anchors_np.reshape((-1,4))

		# num_in < FS_im*FS_im*num_anchors # [num_in,4]
		inside_ind, inside_anchors = self.filter_box_inside(im,featuremap_anchors_flatten) # the anchor box inside the image
		

		# anchor labels is in {1,-1,0}, -1 means ignore
		# N = num_in
		# [N], [N,4] # only the fg anchor has box value
		anchor_labels,anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# fill back to [fs,fs,num_anchor,4]
		# all anchor outside box is ignored (-1)

		featuremap_labels = -np.ones((featureh * featurew*config.num_anchors,),dtype='int32')
		featuremap_labels[inside_ind] = anchor_labels
		featuremap_labels = featuremap_labels.reshape((featureh,featurew,config.num_anchors))

		featuremap_boxes = np.zeros((featureh * featurew*config.num_anchors,4),dtype='float32')
		featuremap_boxes[inside_ind,:] = anchor_boxes
		featuremap_boxes = featuremap_boxes.reshape((featureh,featurew,config.num_anchors,4))

		return featuremap_labels,featuremap_boxes

	def get_multilevel_rpn_anchor_input(self,im,boxes):

		config = self.config

		boxes = boxes.copy()

		anchors_per_level = self.get_all_anchors_fpn() # get anchor for each (anchor_stride,anchor_size) pair
		flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
		all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
		# some image may not be resized to max size, could be shorter edge size
		inside_ind, inside_anchors = self.filter_box_inside(im,all_anchors_flatten)
		# given all these anchors, given the ground truth box, and their iou to each anchor, get the label to be 1 or 0.
		anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# map back to all_anchors, then split to each level
		num_all_anchors = all_anchors_flatten.shape[0]
		all_labels = -np.ones((num_all_anchors, ), dtype='int32')
		all_labels[inside_ind] = anchor_labels
		all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
		all_boxes[inside_ind] = anchor_gt_boxes

		start = 0
		multilevel_inputs = []

		# put back to list for each level

		for level_anchor in anchors_per_level:
			assert level_anchor.shape[2] == len(config.anchor_ratios)
			anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
			num_anchor_this_level = np.prod(anchor_shape)
			end = start + num_anchor_this_level
			multilevel_inputs.append(
				(all_labels[start: end].reshape(anchor_shape),
				all_boxes[start:end, :].reshape(anchor_shape + (4,))))
			start = end

		assert end == num_all_anchors, ("num all anchors:%s, end:%s"%(num_all_anchors,end))
		return multilevel_inputs




	def get_anchor_labels(self,anchors,gt_boxes):
		config = self.config

		# return max_num of index for labels equal val
		def filter_box_label(labels, val, max_num):
			cur_inds = np.where(labels == val)[0]
			if len(cur_inds) > max_num:
				disable_inds = np.random.choice(cur_inds,size=(len(cur_inds) - max_num),replace=False)
				labels[disable_inds] = -1
				cur_inds = np.where(labels == val)[0]
			return cur_inds

		NA,NB = len(anchors),len(gt_boxes)
		assert NB > 0

		#bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
		#box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
		box_ious = np_iou(anchors, gt_boxes)

		#print box_ious.shape #(37607,7)

		#NA, each anchors max iou to any gt box, and the max gt box's index [0,NB-1]
		iou_argmax_per_anchor = box_ious.argmax(axis=1)
		iou_max_per_anchor = box_ious.max(axis=1)

		# 1 x NB, each gt box's max iou to any anchor boxes
		#iou_max_per_gt = box_ious.max(axis=1,keepdims=True) 
		#print iou_max_per_gt # all zero?
		iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

		# NA x 1? True for anchors that cover all the gt boxes
		anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

		anchor_labels = -np.ones((NA,),dtype='int32')

		anchor_labels[anchors_with_max_iou_per_gt] = 1
		anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
		anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

		# cap the number of fg anchor and bg anchor
		target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

		# set the label==1 to -1 if the number exceeds
		fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

		#assert len(fg_inds) > 0
		old_num_bg = np.sum(anchor_labels == 0)
		if old_num_bg == 0:
			raise Exception("No valid background for RPN!")

		# the rest of 256 is negative
		target_num_bg = config.rpn_batch_per_im - len(fg_inds)

		# set some label to -1 if exceeds
		filter_box_label(anchor_labels,0,target_num_bg)

		# only the fg anchor_boxes are filled with the corresponding gt_box
		anchor_boxes = np.zeros((NA,4),dtype='float32')
		anchor_boxes[fg_inds,:] = gt_boxes[iou_argmax_per_anchor[fg_inds],:]
		return anchor_labels, anchor_boxes

class Mask_RCNN_FPN_Act2():
	def __init__(self,config):

		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model

		self.config = config

		self.num_class = config.num_class
		self.num_act_class = config.num_act_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		# for training
		self.anchor_labels = []
		self.anchor_boxes = []
		num_anchors = len(config.anchor_ratios)
		for k in xrange(len(config.anchor_strides)):
			self.anchor_labels.append(tf.placeholder(tf.int32,[None, None, num_anchors],name="anchor_labels_lvl%s"%(k+2)))
			self.anchor_boxes.append(tf.placeholder(tf.float32,[None, None, num_anchors,4],name="anchor_boxes_lvl%s"%(k+2)))

		self.gt_boxes = tf.placeholder(tf.float32,[None, 4],name="gt_boxes")
		self.gt_labels = tf.placeholder(tf.int64,[None,],name="gt_labels")

		# multi class labels, including 0-indexed BG class
		self.act_single_labels = tf.placeholder(tf.uint8,[None,config.num_act_single_class],name="act_single_labels")
		# [K,K,num_act_class]
		self.act_pair_labels = tf.placeholder(tf.uint8,[None,None,config.num_act_pair_class],name="act_pair_labels")

		self.gt_mask = tf.placeholder(tf.uint8,[None, None, None],name="gt_masks") # H,W,v -> {0,1}

		# the following will be added in the build_forward and loss
		self.logits = None
		self.yp = None
		self.loss = None

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		self.multilevel_anchors = []
		with tf.name_scope("fpn_anchors"):#,tf.device("/cpu:0"):
			#fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor box coordinates for a given max_size image,
			# so for 1920 x 1920 image, 1290/16 = 120, so (120,120,NA,4) box, NA is scale*ratio boxes
			self.multilevel_anchors = self.get_all_anchors_fpn()


		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)

			
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			p_image = p_image*(1.0/255)


			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image

	def get_all_anchors_fpn(self):
		config = self.config
		anchors = []
		assert len(config.anchor_strides) == len(config.anchor_sizes)
		for stride, size in zip(config.anchor_strides, config.anchor_sizes):
			anchors_np = get_all_anchors(stride=stride,sizes=[size],ratios=config.anchor_ratios,max_size=config.max_size) 
		
			anchors.append(anchors_np)
		return anchors
		

	def slice_feature_and_anchors(self,image_shape2d,p23456,anchors):
		# anchors is the numpy anchors for different levels
		config = self.config
		# the anchor labels and boxes are grouped into 
		gt_anchor_labels = self.anchor_labels
		gt_anchor_boxes = self.anchor_boxes
		self.sliced_anchor_labels = []
		self.sliced_anchor_boxes = []
		for i,stride in enumerate(config.anchor_strides):
			with tf.name_scope("FPN_slice_lvl%s"%(i)):
				if i<3:
					# Images are padded for p5, which are too large for p2-p4.
					pi = p23456[i]
					target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * (1.0 / stride)))


					p23456[i] = tf.slice(pi, [0,0,0,0],tf.concat([[-1,-1], target_shape], axis=0))
					p23456[i].set_shape([1, pi.shape[1], None,None])

				shape2d = tf.shape(p23456[i])[2:] # h,W
				slice3d = tf.concat([shape2d, [-1]],axis=0)
				slice4d = tf.concat([shape2d, [-1,-1]],axis=0)

				anchors[i] = tf.slice(anchors[i], [0,0,0,0], slice4d)
				self.sliced_anchor_labels.append(tf.slice(gt_anchor_labels[i], [0, 0, 0], slice3d))
				self.sliced_anchor_boxes.append(tf.slice(gt_anchor_boxes[i], [0, 0, 0, 0], slice4d))


	def generate_fpn_proposals(self, multilevel_anchors, multilevel_label_logits,multilevel_box_logits, image_shape2d):
		config = self.config
		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_anchors)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_label_logits)
		all_boxes = []
		all_scores = []
		fpn_nms_topk = config.rpn_train_post_nms_topk if config.is_train else config.rpn_test_post_nms_topk
		for lvl in xrange(num_lvl):
			with tf.name_scope("Lvl%s"%(lvl+2)):
				anchors = multilevel_anchors[lvl]
				pred_boxes_decoded = decode_bbox_target(multilevel_box_logits[lvl], anchors,decode_clip=config.bbox_decode_clip)

				proposal_boxes, proposal_scores = generate_rpn_proposals(tf.reshape(pred_boxes_decoded, [-1,4]), tf.reshape(multilevel_label_logits[lvl], [-1]), image_shape2d, config,pre_nms_topk=fpn_nms_topk)
				all_boxes.append(proposal_boxes)
				all_scores.append(proposal_scores)


		proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
		proposal_scores = tf.concat(all_scores, axis=0)  # n
		proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
		proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
		proposal_boxes = tf.gather(proposal_boxes, topk_indices)
		return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')

	# based on box sizes
	def fpn_map_rois_to_levels(self, boxes):

		def tf_area(boxes):
			x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
			return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

		sqrtarea = tf.sqrt(tf_area(boxes))
		level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))
		# RoI levels range from 2~5 (not 6)
		level_ids = [ 
			tf.where(level <= 2),
			tf.where(tf.equal(level, 3)),
			tf.where(tf.equal(level, 4)),
			tf.where(level >= 5)]

		level_ids = [tf.reshape(x, [-1], name='roi_level%s_id'%(i + 2)) for i, x in enumerate(level_ids)]
		num_in_levels = [tf.size(x, name='num_roi_level%s'%(i + 2)) for i, x in enumerate(level_ids)]

		level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
		return level_ids, level_boxes


	# output_shape is the output feature HxW
	def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
		config = self.config
		assert len(features) == 4
		# Reassign rcnn_boxes to levels # based on box area size
		level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
		all_rois = []

		# Crop patches from corresponding levels
		for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
			with tf.name_scope('roi_level%s'%(i + 2)):
				boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])
				all_rois.append(roi_align(featuremap, boxes_on_featuremap, output_shape))

		# this can fail if using TF<=1.8 with MKL build
		all_rois = tf.concat(all_rois, axis=0)  # NCHW
		# Unshuffle to the original order, to match the original samples
		level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
		level_id_invert_perm = tf.invert_permutation(level_id_perm)
		all_rois = tf.gather(all_rois, level_id_invert_perm)
		return all_rois


	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		multilevel_anchors = self.multilevel_anchors # a list of numpy anchors, not sliced

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		
		c2345 = resnet_fpn_backbone(image,config.resnet_num_block,resolution_requirement=config.fpn_resolution_requirement,tf_pad_reverse=config.new_tensorpack_model,freeze=config.freeze)

		# freeze backbone
		#c2,c3,c4,c5 = c2345
		#c2345 = tf.stop_gradient(c2),tf.stop_gradient(c3),tf.stop_gradient(c4),tf.stop_gradient(c5)

		# include lateral 1x1 conv and final 3x3 conv
		p23456 = fpn_model(c2345,num_channel=config.fpn_num_channel,scope="fpn")



		# given the numpy anchor for each stride, 
		# slice the anchor box and label against the feature map size on each level. Again?
		self.slice_feature_and_anchors(image_shape2d,p23456,multilevel_anchors)
		# now multilevel_anchors are sliced and tf type
		# added sliced gt anchor labels and boxes
		# so we have each fpn level's anchor boxes, and the ground truth anchor boxes & labels if training

		# given [1,256,FS,FS] feature, each level got len(anchor_ratios) anchor outputs
		rpn_outputs = [self.rpn_head(pi, config.fpn_num_channel, len(config.anchor_ratios), data_format="NCHW",scope="rpn") for pi in p23456]
		multilevel_label_logits = [k[0] for k in rpn_outputs]
		multilevel_box_logits = [k[1] for k in rpn_outputs]

		proposal_boxes, proposal_scores = self.generate_fpn_proposals(multilevel_anchors, multilevel_label_logits, multilevel_box_logits, image_shape2d)

		if config.is_train:
			gt_boxes = self.gt_boxes
			gt_labels = self.gt_labels
			# for training, use gt_box and some proposal box as pos and neg
			# rcnn_sampled_boxes [N_FG+N_NEG,4]
			# fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
			rcnn_boxes, rcnn_labels, fg_act_single_labels,fg_act_pair_labels,fg_inds_wrt_gt = sample_fast_rcnn_targets_plus_act(proposal_boxes, gt_boxes,gt_labels, self.act_single_labels,self.act_pair_labels, config=config)
			#fg_act_single_labels [N_FG, num_act]
		else:
			rcnn_boxes = proposal_boxes

		# fast rcnn for object
		# NxCx7x7 # (?, 256, 7, 7)
		roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4],rcnn_boxes, 7)
		# (N,81) ,(N, 80,4)
		# activity head, do single box classification and pari box classification
		# [N, 36?], [N,N,36?]
		# activity utilize object head output
		fastrcnn_label_logits, fastrcnn_box_logits,act_single_label_logits = self.fastrcnn_2fc_act_head(roi_feature_fastrcnn,config.num_class,num_act_single_class=config.num_act_single_class, boxes=rcnn_boxes,scope="fastrcnn")

		if config.is_train:
			rpn_label_loss, rpn_box_loss = self.multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
			
			losses = [rpn_label_loss, rpn_box_loss]
			# object fastrcnn loss
			#---------------- get fast rcnn loss

			# rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
			# [K,2],
			fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])

			# for training, maskRCNN only apply on positive box
			# [N_FG, num_class, 14, 14]

			# [N_FG, 4]
			# sampled boxes are at least iou with a gt_boxes
			fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
			fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

			# [N_FG, 4] # each proposal box assigned gt box, may repeat
			# each RPN outputed foreground box's corresponding groundtruth box
			matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

			# from outputed foreground box to corresponding groundtruth boxes
			encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * tf.constant(config.fastrcnn_bbox_reg_weights) #[10,10,5,5]?

			# the act_single label is only on FG box, so we gather the logit on FG box only as well
			# TODO: select with a fix pos:neg ratio
			fg_act_single_label_logits = tf.gather(act_single_label_logits, fg_inds_wrt_sample)

			# fastrcnn input is fg and bg proposal box, do classification to num_class(include bg) and then regress on fg boxes
			# [N_FG+N_NEG,4] & [N_FG,4] & [N_FG,num_act_class] 
			fastrcnn_label_loss, fastrcnn_box_loss, act_single_label_loss = self.fastrcnn_act_losses(rcnn_labels, fg_act_single_labels, fastrcnn_label_logits,fg_act_single_label_logits, encoded_boxes, fg_fastrcnn_box_logits)

			act_single_label_loss = act_single_label_loss * tf.constant(config.act_loss_weight)
			# ---------------------------------------------------------

			# for debug
			self.rpn_label_loss = rpn_label_loss
			self.rpn_box_loss = rpn_box_loss
			self.fastrcnn_label_loss = fastrcnn_label_loss
			self.fastrcnn_box_loss = fastrcnn_box_loss

			losses.extend([fastrcnn_label_loss, fastrcnn_box_loss,act_single_label_loss])

			self.act_losses = []

			self.act_losses.extend([act_single_label_loss])

			if config.wd is not None:
				wd = wd_cost('.*/W', config.wd,scope="wd_cost")
				losses.append(wd)

			self.loss = tf.add_n(losses,'total_loss')

			# l2loss
		else:

			# get the regressed actual boxes
			# anchor box [K,4] -> [K,num_class - 1, 4] <- box regress logits [K,num_class-1,4]
			anchors = tf.tile(tf.expand_dims(rcnn_boxes,1),[1, config.num_class-1,1])
			decoded_boxes = decode_bbox_target(fastrcnn_box_logits / tf.constant(config.fastrcnn_bbox_reg_weights,dtype=tf.float32), anchors)
			#[K,num_class - 1,4]
			decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")

			label_probs = tf.nn.softmax(fastrcnn_label_logits)

			#[R,2], [R]
			# pred_indices -> [R,2] ,(K_id,cat_id)
			pred_indices, final_probs = self.fastrcnn_predictions(decoded_boxes, label_probs)
			# [R,4]
			final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name="final_boxes")
			# [R] , each is 1-80 catogory
			final_labels = tf.add(pred_indices[:,1],1,name="final_labels")
			
			
			#--------------------------------
			# act_single_label_logits [K, num_act_class]
			# version 1: get the top object boxes' activity classification
			# [R,num_act_class]
			act_single_label_logits_selected = tf.gather(act_single_label_logits,pred_indices[:,0])
			act_single_boxes = final_boxes # this is the same as object box,


			# version 2, select different boxes based on prob
			# get object box from [K,num_class,4] -> [K,4] first
			"""
			#[K]
			each_max_class = tf.argmax(label_probs,axis=1) - 1 # -1 since box is num_class - 1
			#[K,2]
			select_box_indices = tf.stack([tf.range(tf.shape(label_probs)[0]),tf.to_int32(each_max_class)], axis=1)
			# [K,4]
			selected_boxes = tf.gather_nd(decoded_boxes, select_box_indices)

			act_single_label_probs = tf.nn.softmax(act_single_label_logits)
			act_single_pred_indices, act_single_final_probs = self.fastrcnn_predictions_act(selected_boxes, act_single_label_probs)

			# [R,4]
			act_single_boxes = tf.gather(selected_boxes, act_single_pred_indices[:,0], name="final_act_boxes")
			# [R,num_act_class]
			act_single_label_logits_selected = tf.gather(act_single_label_logits,act_single_pred_indices[:,0],name="final_act_logits")
			"""

			# [R,4]
			self.final_boxes = final_boxes
			# [R]
			self.final_labels = final_labels
			self.final_probs = final_probs
			
			# [R, num_act_single-class] # include BG class
			self.act_single_boxes = act_single_boxes
			self.act_single_label_logits = act_single_label_logits_selected
			

	# ----some model component
	# feature map -> [1,1024,FS1,FS2] , FS1 = H/16.0, FS2 = W/16.0
	# channle -> 1024
	def rpn_head(self,featuremap, channel, num_anchors, data_format,scope="rpn"):
		with tf.variable_scope(scope):
			# [1, channel, FS1, FS2] # channel = 1024
			# conv0:W -> [3,3,1024,1024]
			h = conv2d(featuremap,channel,kernel=3,activation=tf.nn.relu,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="conv0")
			# h -> [1,1024(channel),FS1,FS2]

			# 1x1 kernel conv to classification on each grid
			# [1, 1024, FS1, FS2] -> # [1, num_anchors, FS1, FS2]
			label_logits = conv2d(h,num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="class")
			# [1, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
			box_logits = conv2d(h,4*num_anchors,1,data_format=data_format,W_init=tf.random_normal_initializer(stddev=0.01),scope="box")

			# [1,1024,FS1, FS2] -> [FS1, FS2,1024]
			label_logits = tf.squeeze(tf.transpose(label_logits, [0,2,3,1]),0)

			box_shape = tf.shape(box_logits)
			box_logits = tf.transpose(box_logits,[0,2,3,1]) # [1,FS1, FS2,1024*4]
			# [FS1, FS2,1024,4]
			box_logits = tf.reshape(box_logits,[box_shape[2], box_shape[3],num_anchors,4])

			return label_logits,box_logits

	# feature: [K,C,7,7] # feature for each roi
	def fastrcnn_2fc_act_head(self,feature,num_class,num_act_single_class,boxes=None,scope="fastrcnn_head"):
		config = self.config
		dim = config.fpn_frcnn_fc_head_dim # 1024
		initializer = tf.variance_scaling_initializer()

		with tf.variable_scope(scope):
			# dense will reshape to [k,C*7*7] first
			if config.add_relation_nn:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")
				hidden = hidden + relation_network(hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
			else:
				hidden = dense(feature,dim,W_init=initializer,activation=tf.nn.relu,scope="fc6")
				hidden = dense(hidden,dim,W_init=initializer,activation=tf.nn.relu,scope="fc7")

			with tf.variable_scope("outputs"):

				classification = dense(hidden,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]

				if config.new_tensorpack_model:
					box_regression = dense(hidden,num_class*4 ,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					box_regression = tf.reshape(box_regression, (-1, num_class,4))

					box_regression = box_regression[:,1:,:]
					
					box_regression.set_shape([None,num_class-1,4])
				else:
					box_regression = dense(hidden,(num_class -1)*4, W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
					
					box_regression = tf.reshape(box_regression, (-1, num_class-1,4))

			with tf.variable_scope("activity_detection"):
				# single box activity
				# TODO: add spatial attention on the c,7,7 feature
				# TODO: add object class output as feature
				class_feat_dim = 256
				obj_class_feat = dense(classification,class_feat_dim,activation=tf.nn.relu,scope="obj_class_feat")
				single_act_classification = dense(tf.concat([hidden, obj_class_feat],axis=-1),num_act_single_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="act_single_class") # [K,num_act_single_class]

				# TODO: pair box activity
			
			

		return classification,box_regression, single_act_classification


	
	def maskrcnn_up4conv_head(self,feature,num_class,scope="maskrcnn_head"):
		config = self.config
		num_conv = 4 # C4 model this is 0
		l = feature
		with tf.variable_scope(scope):
			for k in xrange(num_conv):
				l = conv2d(l, config.mrcnn_head_dim, kernel=3, activation=tf.nn.relu, data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='truncated_normal'), scope="fcn%s"%(k))

			l = deconv2d(l, config.mrcnn_head_dim, kernel=2, stride=2, activation=tf.nn.relu, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out", distribution='truncated_normal'), scope="deconv")
			l = conv2d(l,num_class-1, kernel=1, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_out",distribution='normal'), scope="conv")
			return l

	# given all proposal box prediction, based on score thres , get final NMS resulting box
	# [K,num_class-1,4] -> decoded_boxes
	# [K,num_class] label_probs
	# each proposal box has prob and box to all class
	# here using nms for each class, -> [R]
	def fastrcnn_predictions(self,boxes, probs, num_class=None, scope="fastrcnn_predictions"):
		with tf.variable_scope(scope):		
			config = self.config
			if num_class is None:
				num_class = config.num_class
		
			assert boxes.shape[1] == num_class - 1,(boxes.shape,num_class)
			assert probs.shape[1] == num_class,(probs.shape[1],num_class)
			# transpose to map_fn along each class
			boxes = tf.transpose(boxes,[1,0,2]) # [num_class-1, K,4]
			# ignored BG class prob
			probs = tf.transpose(probs[:,1:],[1,0]) # [num_class-1, K]

			def f(X):
				prob,box = X # [K], [K,4]
				output_shape = tf.shape(prob)
				# [K]
				ids = tf.reshape(tf.where(prob > config.result_score_thres),[-1])
				prob = tf.gather(prob,ids)
				box = tf.gather(box,ids)
				# NMS
				selection = tf.image.non_max_suppression(box,prob,max_output_size=config.result_per_im,iou_threshold=config.fastrcnn_nms_iou_thres)
				selection = tf.to_int32(tf.gather(ids,selection))
				sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

				mask = tf.sparse_to_dense(sparse_indices=sorted_selection,output_shape=output_shape,sparse_values=True,default_value=False)
				return mask

			# for each catagory get the top K
			# [num_class-1, R]
			masks = tf.map_fn(f, (probs,boxes), dtype=tf.bool, parallel_iterations=10)
			# [R,2], each entry is [cat_id,box_id]
			selected_indices = tf.where(masks)

			probs = tf.boolean_mask(probs,masks)# [num_class-1, K] 

			# topk_indices [num_class-1,result_num]
			topk_probs, topk_indices = tf.nn.top_k(probs, tf.minimum(config.result_per_im,tf.size(probs)),sorted=False)

			# [K,2] <- select [act_num,R] 
			filtered_selection = tf.gather(selected_indices, topk_indices)
			filtered_selection = tf.reverse(filtered_selection, axis=[1],name="filtered")

			# [R,2], [R,]
			return filtered_selection, topk_probs

	# given all proposal box prediction, based on single box activity score thres , get final NMS resulting box
	# [K,4] -> decoded_boxes
	# [K,num_act_class] label_probs, softmaxed
	# 
	def fastrcnn_predictions_act(self,boxes, act_probs, scope="fastrcnn_predictions"):
		with tf.variable_scope(scope):		
			config = self.config
			num_class = config.num_class
			num_act_single_class = config.num_act_single_class
		
			assert boxes.shape[1] == 4,(boxes.shape,num_class)
			assert act_probs.shape[1] == num_act_single_class,(act_probs.shape,num_act_single_class)
			# transpose to map_fn along each class
			boxes = tf.tile(tf.expand_dims(boxes,axis=1),[1,num_act_single_class,1])# [K,4] -> [K,num_act_class, 4]
			boxes = tf.transpose(boxes,[1,0,2]) # [num_act_class, K,4]
			# TODO: ignored BG class prob?
			#act_probs = act_probs[:,1:]
			probs = tf.transpose(act_probs,[1,0]) # [num_act_class, K]

			def f(X):
				prob,box = X # [K], [K,4]
				output_shape = tf.shape(prob)
				# [K]
				ids = tf.reshape(tf.where(prob > config.result_score_thres),[-1])
				prob = tf.gather(prob,ids)
				box = tf.gather(box,ids)
				# NMS
				selection = tf.image.non_max_suppression(box,prob,max_output_size=config.result_per_im,iou_threshold=config.fastrcnn_nms_iou_thres)
				selection = tf.to_int32(tf.gather(ids,selection))
				sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

				mask = tf.sparse_to_dense(sparse_indices=sorted_selection,output_shape=output_shape,sparse_values=True,default_value=False)
				return mask

			# for each catagory get the top K
			# [num_class-1, R]
			masks = tf.map_fn(f, (probs,boxes), dtype=tf.bool, parallel_iterations=10)
			# [R,2], each entry is [cat_id,box_id]
			selected_indices = tf.where(masks)

			probs = tf.boolean_mask(probs,masks)# [num_class-1, K] 

			# topk_indices [num_class-1,result_num]
			topk_probs, topk_indices = tf.nn.top_k(probs, tf.minimum(config.result_per_im,tf.size(probs)),sorted=False)

			# [K,2] <- select [act_num,R] 
			filtered_selection = tf.gather(selected_indices, topk_indices)
			filtered_selection = tf.reverse(filtered_selection, axis=[1],name="filtered")



			# [R,2], [R,]
			return filtered_selection, topk_probs

	# ---- losses
	def maskrcnn_loss(self,mask_logits, fg_labels, fg_target_masks,scope="maskrcnn_loss"):
		with tf.variable_scope(scope):
			# mask_logits: [N_FG, num_cat, 14, 14]
			# fg_labels: [N_FG]
			# fg_target_masks: [N_FG, 14, 14]
			num_fg = tf.size(fg_labels)
			# [N_FG, 2] # these index is used to get the pos cat's logit
			indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1],axis=1)
			# ignore other class's logit
			# [N_FG, 14, 14]
			mask_logits = tf.gather_nd(mask_logits, indices)
			mask_probs = tf.sigmoid(mask_logits)

			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
			loss = tf.reduce_mean(loss, name='maskrcnn_loss')

			return loss


	def multilevel_rpn_losses(self, multilevel_anchors, multilevel_label_logits, multilevel_box_logits, scope="rpn_losses"):
		config = self.config
		sliced_anchor_labels = self.sliced_anchor_labels 
		sliced_anchor_boxes = self.sliced_anchor_boxes

		num_lvl = len(config.anchor_strides)
		assert num_lvl == len(multilevel_label_logits)
		assert num_lvl == len(multilevel_box_logits)
		assert num_lvl == len(multilevel_anchors)

		losses = []
		with tf.variable_scope(scope):
			for lvl in xrange(num_lvl):
				anchors = multilevel_anchors[lvl]
				gt_labels = sliced_anchor_labels[lvl]
				gt_boxes = sliced_anchor_boxes[lvl]

				# get the ground truth T_xywh
				encoded_gt_boxes = encode_bbox_target(gt_boxes, anchors)

				label_loss, box_loss = self.rpn_losses(gt_labels, encoded_gt_boxes, multilevel_label_logits[lvl], multilevel_box_logits[lvl],scope="level%s"%(lvl+2))
				losses.extend([label_loss,box_loss])

			total_label_loss = tf.add_n(losses[::2], name='label_loss')
			total_box_loss = tf.add_n(losses[1::2], name='box_loss')

		return total_label_loss, total_box_loss

			
	def rpn_losses(self, anchor_labels, anchor_boxes, label_logits, box_logits,scope="rpn_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
			# label_logits [FS,FS,num_anchors] [7,7,1024]
			# box_logits [FS,FS,num_anchors,4] [7.7,1024,4]
			
			#with tf.device("/cpu:0"):
			valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1)) # 1,0|pos/neg
			pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
			nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
			nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')

			# [K1]

			valid_anchor_labels = tf.boolean_mask(anchor_labels,valid_mask)

			# [K2]
			valid_label_logits = tf.boolean_mask(label_logits, valid_mask)


			placeholder = 0.

			# label loss for all valid anchor box
			if config.focal_loss:
				label_loss = focal_loss(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))
			else:
				label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=valid_label_logits,labels=tf.to_float(valid_anchor_labels))

				label_loss = tf.reduce_mean(label_loss,name="label_loss")

				label_loss = tf.reduce_sum(label_loss) * (1. / config.rpn_batch_per_im)

			label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

			# box loss for positive anchor
			pos_anchor_boxes = tf.boolean_mask(anchor_boxes,pos_mask)
			pos_box_logits = tf.boolean_mask(box_logits,pos_mask)

			delta = 1.0/9

			# the smooth l1 loss
			box_loss = tf.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta, reduction=tf.losses.Reduction.SUM) / delta

			#box_loss = tf.div(box_loss, tf.cast(nr_valid, tf.float32),name='box_loss')
			box_loss = box_loss * (1. / config.rpn_batch_per_im)
			box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')


			return label_loss, box_loss

	def fastrcnn_act_losses(self, labels, act_single_labels, label_logits, act_single_label_logits, fg_boxes, fg_box_logits,scope="fastrcnn_act_losses"):
		config = self.config
		with tf.variable_scope(scope):
			# label -> label for roi [N_FG + N_NEG]
			# act is already a multi-hot labels
			# act_single_labels -> [N_FG,num_act_single_class] #some N_FG is already BG 
			# label_logits [N_FG + N_NEG,num_class]
			# fg_boxes_logits -> [N_FG,num_class-1,4], [N_FG,1,4] for cascade rcnn

			# so the label is int [0-num_class], 0 being background

			fg_inds = tf.where(labels > 0)[:,0] # same as reshaping into [-1]

			if config.focal_loss:
				onehot_label = tf.one_hot(labels,label_logits.get_shape()[-1])

				# here uses sigmoid
				label_loss = focal_loss(logits=label_logits,labels=tf.to_float(onehot_label))

				act_single_label_loss = focal_loss(logits=act_single_label_logits,labels=tf.to_float(act_single_labels))
			else:
				label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)

				label_loss = tf.reduce_mean(label_loss, name="label_loss")

				# multi class
				# we only get loss on the fg obj boxes, 
				# act_single_labels [N_FG, num_act_class]
				# act_single_label_logits [N_FG, num_act_class]
				# since still too many BG class in N_FG, we random sample  for loss
				# -> [N]
				act_fg_mask = tf.equal(act_single_labels[:,0],0) # act_single_labels[:,0]==0 will fail
				act_fg_inds = tf.reshape(tf.where(act_fg_mask),[-1]) # [K_FG] # index of fg_mask true element

				act_num_fg = tf.minimum(int(config.fastrcnn_batch_per_im * config.fastrcnn_act_fg_ratio),tf.size(act_fg_inds))
				# during train time, each time random sample
				act_fg_inds = tf.random_shuffle(act_fg_inds)[:act_num_fg]# so the pos box is at least > fg_thres iou

				act_bg_inds = tf.reshape(tf.where(tf.logical_not(act_fg_mask)),[-1])
				act_num_bg = tf.minimum(config.fastrcnn_batch_per_im - act_num_fg,tf.size(act_bg_inds))
				act_bg_inds = tf.random_shuffle(act_bg_inds)[:act_num_bg]

				act_all_indices = tf.concat([act_fg_inds,act_bg_inds],axis=0)

				act_single_labels_selected = tf.gather(act_single_labels,act_all_indices)
				act_single_label_logits_selected = tf.gather(act_single_label_logits,act_all_indices)

				act_single_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(act_single_labels_selected), logits=act_single_label_logits_selected)
				act_single_label_loss = tf.reduce_mean(act_single_label_loss, name="act_single_label_loss")

			fg_labels = tf.gather(labels, fg_inds) # [N_FG]

			num_fg = tf.size(fg_inds) # N_FG
			if int(fg_box_logits.shape[1]) > 1:
				# [N_FG](class_indexes) -> [N_FG, 2]
				indices = tf.stack([tf.range(num_fg),tf.to_int32(fg_labels) - 1], axis=1)
				# gather the logits from [N_FG,num_class-1, 4] to [N_FG,4], only the gt class's logit
				fg_box_logits = tf.gather_nd(fg_box_logits, indices)
			else:
				fg_box_logits = tf.reshape(fg_box_logits, [-1, 4]) # class agnostic for cascade rcnn
			box_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

			# /  N_FG + N_NEG ?
			box_loss = tf.truediv(box_loss, tf.to_float(tf.shape(labels)[0]),name='box_loss')

			return label_loss, box_loss, act_single_label_loss


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,batch,is_train=False):

		#{"imgs":[],"gt":[]}
		config = self.config
		
		N = len(batch.data['imgs'])

		assert N == 1 # only 1 image for now

		image = batch.data['imgs'][0]

		feed_dict = {}

		if batch.data.has_key("imgdata"):
			image = batch.data['imgdata'][0]
		else:
			image = cv2.imread(image,cv2.IMREAD_COLOR)
			assert image is not None,image
			image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		# resize image, boxes
		short_edge_size = config.short_edge_size
		if config.scale_jitter and is_train:
			short_edge_size = random.randint(config.short_edge_size_min,config.short_edge_size_max)

		if batch.data.has_key("resized_image"):
			resized_image = batch.data['resized_image'][0]
		else:
			resized_image = resizeImage(image,short_edge_size,config.max_size)
		newh,neww = resized_image.shape[:2]

		if is_train:
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box
			labels = anno['labels']
			actSingleLabels = anno['actSingleLabels_npy'] # [uint8]
			
			assert len(labels) == len(o_boxes) == len(actSingleLabels)

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2

			# random horizontal flip
			# no flip for surveilance video?
			if config.flip_image:
				prob = 0.5
				rand = random.random()
				if rand > prob:
					resized_image = cv2.flip(resized_image,1) # 1 for horizontal
					#boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
					boxes[:,0] = neww - boxes[:,0]
					boxes[:,0,:] = boxes[:,0,::-1]# (x_min will be x_max after flip)				

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)

			# visualize?
			if config.vis_pre:
				label_names = [config.classId_to_class[i] for i in labels]
				o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
				boxes_x1x2 = np.asarray([box for box in boxes])
				ori_vis = draw_boxes(image,o_boxes_x1x2,labels=label_names)
				new_vis = draw_boxes(resized_image,boxes_x1x2,labels=label_names)
				imgname = os.path.splitext(os.path.basename(batch.data['imgs'][0]))[0]
				cv2.imwrite("%s.ori.jpg"%os.path.join(config.vis_path,imgname),ori_vis)
				cv2.imwrite("%s.prepro.jpg"%os.path.join(config.vis_path,imgname),new_vis)
				print "viz saved in %s"%config.vis_path
				sys.exit()

			# get rpn anchor labels
			# [fs_im,fs_im,num_anchor,4]
			
			#multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(resized_image, boxes)
			# RPN target is both the obj boxes and activity boxes?
			multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(resized_image, boxes)

			multilevel_anchor_labels = [l for l,b in multilevel_anchor_inputs]
			multilevel_anchor_boxes = [b for l,b in multilevel_anchor_inputs]
			assert len(multilevel_anchor_labels) == len(multilevel_anchor_boxes) == len(self.anchor_labels) == len(self.anchor_boxes), (len(multilevel_anchor_labels), len(multilevel_anchor_boxes), len(self.anchor_labels),len(self.anchor_boxes) )

			for pl_labels,pl_boxes,in_labels,in_boxes in zip(self.anchor_labels,self.anchor_boxes,multilevel_anchor_labels, multilevel_anchor_boxes):

				feed_dict[pl_labels] = in_labels
				feed_dict[pl_boxes] = in_boxes

			assert len(boxes) > 0

			feed_dict[self.gt_boxes] = boxes
			feed_dict[self.gt_labels] = labels

			feed_dict[self.act_single_labels] = actSingleLabels

		else:
			
			pass

		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

	def get_feed_dict_forward(self,imgdata):
		feed_dict = {}

		feed_dict[self.image] = imgdata

		feed_dict[self.is_train] = False

		return feed_dict

	# anchor related function for training--------------------

	def filter_box_inside(self, im, boxes):
		h, w = im.shape[:2]
		indices = np.where(
			(boxes[:,0] >= 0) &
			(boxes[:,1] >= 0) &
			(boxes[:,2] <= w) &
			(boxes[:,3] <= h)  
		)[0]
		return indices, boxes[indices,:]
	# for training, given image and box, get anchor box labels
	# [fs_im,fs_im,num_anchor,4] # not fs,
	def get_rpn_anchor_input(self,im,boxes):
		

		config = self.config

		boxes = boxes.copy()

		# [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
		all_anchors_np = np.copy(get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size))

		h,w = im.shape[:2]

		# so image may be smaller than the full anchor size
		#featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
		anchorH, anchorW = all_anchors_np.shape[:2]
		featureh, featurew = anchorH, anchorW

		# [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
		#featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
		#print featuremap_anchors.shape #(46,83,15,4)
		#featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))
		featuremap_anchors_flatten = all_anchors_np.reshape((-1,4))

		# num_in < FS_im*FS_im*num_anchors # [num_in,4]
		inside_ind, inside_anchors = self.filter_box_inside(im,featuremap_anchors_flatten) # the anchor box inside the image
		

		# anchor labels is in {1,-1,0}, -1 means ignore
		# N = num_in
		# [N], [N,4] # only the fg anchor has box value
		anchor_labels,anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# fill back to [fs,fs,num_anchor,4]
		# all anchor outside box is ignored (-1)

		featuremap_labels = -np.ones((featureh * featurew*config.num_anchors,),dtype='int32')
		featuremap_labels[inside_ind] = anchor_labels
		featuremap_labels = featuremap_labels.reshape((featureh,featurew,config.num_anchors))

		featuremap_boxes = np.zeros((featureh * featurew*config.num_anchors,4),dtype='float32')
		featuremap_boxes[inside_ind,:] = anchor_boxes
		featuremap_boxes = featuremap_boxes.reshape((featureh,featurew,config.num_anchors,4))

		return featuremap_labels,featuremap_boxes

	def get_multilevel_rpn_anchor_input(self,im,boxes):

		config = self.config

		boxes = boxes.copy()

		anchors_per_level = self.get_all_anchors_fpn() # get anchor for each (anchor_stride,anchor_size) pair
		flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
		all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
		# some image may not be resized to max size, could be shorter edge size
		inside_ind, inside_anchors = self.filter_box_inside(im,all_anchors_flatten)
		# given all these anchors, given the ground truth box, and their iou to each anchor, get the label to be 1 or 0.
		anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

		# map back to all_anchors, then split to each level
		num_all_anchors = all_anchors_flatten.shape[0]
		all_labels = -np.ones((num_all_anchors, ), dtype='int32')
		all_labels[inside_ind] = anchor_labels
		all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
		all_boxes[inside_ind] = anchor_gt_boxes

		start = 0
		multilevel_inputs = []

		# put back to list for each level

		for level_anchor in anchors_per_level:
			assert level_anchor.shape[2] == len(config.anchor_ratios)
			anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
			num_anchor_this_level = np.prod(anchor_shape)
			end = start + num_anchor_this_level
			multilevel_inputs.append(
				(all_labels[start: end].reshape(anchor_shape),
				all_boxes[start:end, :].reshape(anchor_shape + (4,))))
			start = end

		assert end == num_all_anchors, ("num all anchors:%s, end:%s"%(num_all_anchors,end))
		return multilevel_inputs




	def get_anchor_labels(self,anchors,gt_boxes):
		config = self.config

		# return max_num of index for labels equal val
		def filter_box_label(labels, val, max_num):
			cur_inds = np.where(labels == val)[0]
			if len(cur_inds) > max_num:
				disable_inds = np.random.choice(cur_inds,size=(len(cur_inds) - max_num),replace=False)
				labels[disable_inds] = -1
				cur_inds = np.where(labels == val)[0]
			return cur_inds

		NA,NB = len(anchors),len(gt_boxes)
		assert NB > 0

		#bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
		#box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
		box_ious = np_iou(anchors, gt_boxes)

		#print box_ious.shape #(37607,7)

		#NA, each anchors max iou to any gt box, and the max gt box's index [0,NB-1]
		iou_argmax_per_anchor = box_ious.argmax(axis=1)
		iou_max_per_anchor = box_ious.max(axis=1)

		# 1 x NB, each gt box's max iou to any anchor boxes
		#iou_max_per_gt = box_ious.max(axis=1,keepdims=True) 
		#print iou_max_per_gt # all zero?
		iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

		# NA x 1? True for anchors that cover all the gt boxes
		anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

		anchor_labels = -np.ones((NA,),dtype='int32')

		anchor_labels[anchors_with_max_iou_per_gt] = 1
		anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
		anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

		# cap the number of fg anchor and bg anchor
		target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

		# set the label==1 to -1 if the number exceeds
		fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

		#assert len(fg_inds) > 0
		old_num_bg = np.sum(anchor_labels == 0)
		if old_num_bg == 0:
			raise Exception("No valid background for RPN!")

		# the rest of 256 is negative
		target_num_bg = config.rpn_batch_per_im - len(fg_inds)

		# set some label to -1 if exceeds
		filter_box_label(anchor_labels,0,target_num_bg)

		# only the fg anchor_boxes are filled with the corresponding gt_box
		anchor_boxes = np.zeros((NA,4),dtype='float32')
		anchor_boxes[fg_inds,:] = gt_boxes[iou_argmax_per_anchor[fg_inds],:]
		return anchor_labels, anchor_boxes

class Mask_RCNN_boxfeat():
	def __init__(self,config):

		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model
		#is_training = False # this is getting feature so never train
		assert is_training is False

		self.config = config

		self.num_class = config.num_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		self.boxes = tf.placeholder(tf.float32,[None, 4],name="boxes")

		# TODO: input mask and extract feature from mask?
		#self.mask = tf.placeholder(tf.uint8,[None, None, None],name="gt_masks") # H,W,v -> {0,1}

		# the following will be added in the build_forward and loss
		self.feature = None
		self.label_probs = None

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		with tf.name_scope("anchors"):#,tf.device("/cpu:0"):
			fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor for a given config
			all_anchors_np = get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size) 

			all_anchors = tf.constant(all_anchors_np, name="all_anchors",dtype=tf.float32)
			fm_anchors = tf.slice(all_anchors, [0,0,0,0], tf.stack([fm_h,fm_w,-1,-1]),name="fm_anchors")
			self.fm_anchors = fm_anchors

		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)
			p_image = p_image*(1.0/255)
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image
		

	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		fm_anchors = self.fm_anchors # [FS,FS,num_anchors,4] # all posible anchor position for this image

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		featuremap = pretrained_resnet_conv4(image, config.resnet_num_block[:3])
		# feat -> [1,1024,image_h/16,image_w/16]
		#self.featuremap = featuremap

		boxes = self.boxes # [K,4]
		
		# testing use all proposal boxes
		# [K,4] 
		boxes_on_featuremap = boxes * (1.0 / config.anchor_stride)

		# given the boxes, get the fixed size features
		# [K,1024,14,14]
		roi_resized = roi_align(featuremap,boxes_on_featuremap,14)

		# [K,2048,7,7]
		feature_fastrcnn = resnet_conv5(roi_resized,config.resnet_num_block[-1])
		self.label_probs = None
		if config.boxclass:
			fastrcnn_label_logits = self.fastrcnn_head(feature_fastrcnn,config.num_class,scope="fastrcnn")

			# [K,num_class]
			label_probs = tf.nn.softmax(fastrcnn_label_logits)
			
			self.label_probs = label_probs # [K,num_class]

		if config.avg_feat:
			feature_fastrcnn = tf.reduce_mean(feature_fastrcnn,[2,3])

		self.feature = feature_fastrcnn # [K,2048,7,7] / [K,2048]






	# ----some model component
	# feature: [K,2048,7,7] # feature for each roi
	def fastrcnn_head(self,feature,num_class,scope="fastrcnn_head"):
		with tf.variable_scope(scope):
			# [K,2048,7,7] -> [K,2048]
			# global avg pooling
			feature = tf.reduce_mean(feature,axis=[2,3],name="output")
			classification = dense(feature,num_class,W_init=tf.random_normal_initializer(stddev=0.01),scope="class") # [K,num_class]
			# no need for box regression, we are just getting box feature
			#box_regression = dense(feature,(num_class -1)*4,W_init=tf.random_normal_initializer(stddev=0.001),scope="box")
			#box_regression = tf.reshape(box_regression, (-1, num_class-1,4))

			return classification#,box_regression


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,imgpath,boxes,is_train=False):

		#{"imgs":[],"gt":[]}
		config = self.config

		feed_dict = {}

		image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
		assert image is not None,image
		image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		
		resized_image = resizeImage(image,config.short_edge_size,config.max_size)
		newh,neww = resized_image.shape[:2]

		# resize boxes
		# boxes # (x1,y1,x2,y2) # [K,4]
		boxes = boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
		boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
		boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
		boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2
		boxes = boxes.reshape((-1,4))
		boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)

		feed_dict[self.boxes] = boxes

		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

# given box input and produce mask
class Mask_RCNN_givenbox():
	def __init__(self,config):

		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model
		#is_training = False # this is getting feature so never train
		assert is_training is False

		self.config = config

		self.num_class = config.num_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		self.boxes = tf.placeholder(tf.float32,[None, 4],name="boxes")
		self.box_labels = tf.placeholder(tf.int32,[None],name="box_labels")
		self.box_probs = tf.placeholder(tf.float32,[None],name="box_probs")

		self.build_preprocess()
		self.build_forward()


	# get feature map anchor and preprocess image
	def build_preprocess(self):
		config = self.config
		image = self.image

		# get feature map anchors first
		# slower if put on cpu # 1.5it/s vs 1.2it/s
		with tf.name_scope("anchors"):#,tf.device("/cpu:0"):
			fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1] // config.anchor_stride

			# all posible anchor for a given config
			all_anchors_np = get_all_anchors(stride=config.anchor_stride,sizes=config.anchor_sizes,ratios=config.anchor_ratios,max_size=config.max_size) 

			all_anchors = tf.constant(all_anchors_np, name="all_anchors",dtype=tf.float32)
			fm_anchors = tf.slice(all_anchors, [0,0,0,0], tf.stack([fm_h,fm_w,-1,-1]),name="fm_anchors")
			self.fm_anchors = fm_anchors

		bgr = True # cv2 load image is bgr
		p_image = tf.expand_dims(image,0) #[1,H,W,C]
		#print image.get_shape()
		#sys.exit()
		with tf.name_scope("image_preprocess"):#,tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)
			p_image = p_image*(1.0/255)
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std,dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])
		self.p_image = p_image
		

	def build_forward(self):
		config = self.config
		image = self.p_image # [1, C, H, W]
		image_shape2d = tf.shape(image)[2:]
		fm_anchors = self.fm_anchors # [FS,FS,num_anchors,4] # all posible anchor position for this image

		# the feature map shared by RPN and fast RCNN
		# TODO: fix the batch norm mess 
		# TODO: fix global param like data_format and 
		# [1,C,FS,FS]
		featuremap = pretrained_resnet_conv4(image, config.resnet_num_block[:3])
		# feat -> [1,1024,image_h/16,image_w/16]
		#self.featuremap = featuremap

		# [R,4]
		final_boxes = self.boxes
		# [R] , each is 1-80 catogory
		final_labels = self.box_labels

		assert config.add_mask
		def f1():
			# get mask prediction
			# use the final box
			roi_resized = roi_align(featuremap, final_boxes*(1.0/config.anchor_stride),14)
			feature_maskrcnn = resnet_conv5(roi_resized,config.resnet_num_block[-1])
			# [R, num_class-1, 14, 14]
			mask_logits = self.maskrcnn_head(feature_maskrcnn, config.num_class,scope='maskrcnn')

			# get only the predict class's mask
			# [num_class-1,2] -> each is (1-R, #label_class)
			# since the labels is 1-80, here - 1 to be 0-79 index
			indices = tf.stack([tf.range(tf.size(final_labels)),tf.to_int32(final_labels)-1],axis=1)
			# [R,14,14]
			final_mask_logits = tf.gather_nd(mask_logits,indices)
			final_masks = tf.sigmoid(final_mask_logits)
			return final_masks

		final_masks = tf.cond(tf.size(final_boxes) > 0, f1, lambda: tf.zeros([0, 14, 14]))
		# [R,14,14]
		self.final_masks = final_masks

		# [R,4]
		self.final_boxes = final_boxes
		# [R]
		self.final_labels = final_labels
		self.final_probs = self.box_probs
			


	# ----some model component

	def maskrcnn_head(self,feature,num_class,scope="maskrcnn_head"):
		with tf.variable_scope(scope):
			# feature: [K, 2048, 7, 7] # K box
			# num_class: num_cat + 1 [background]
			# return: [K, num_cat, 14, 14]
			l = deconv2d(feature, 256, kernel=2, stride=2, activation=tf.nn.relu,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_in",distribution='normal'), scope="deconv")
			l = conv2d(l,num_class-1,kernel=1,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=2.0,mode="fan_in",distribution='normal'), scope="conv")
			return l


	# given the image path, and the label for it
	# preprocess
	def get_feed_dict(self,imgpath,boxes,box_labels,box_probs):

		#{"imgs":[],"gt":[]}
		config = self.config

		feed_dict = {}

		image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
		assert image is not None,image
		image = image.astype("float32")
		h,w = image.shape[:2] # original width/height

		
		resized_image = resizeImage(image,config.short_edge_size,config.max_size)

		newh,neww = resized_image.shape[:2]

		# resize boxes
		# boxes # (x1,y1,x2,y2) # [K,4]
		boxes = boxes[:,[0,2,1,3]] #(x1,x2,y1,y2)
		boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
		boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
		boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2
		boxes = boxes.reshape((-1,4))
		boxes = boxes[:,[0,2,1,3]] #(x1,y1,x2,y2)

		feed_dict[self.boxes] = boxes
		feed_dict[self.box_labels] = box_labels
		feed_dict[self.box_probs] = box_probs

		feed_dict[self.image] = resized_image

		# return the original shape to recover original box and mask
		scale = (resized_image.shape[0]*1.0/image.shape[0] + resized_image.shape[1]*1.0/image.shape[1])/2.0
		
		return feed_dict,image.shape[:2],scale


		
