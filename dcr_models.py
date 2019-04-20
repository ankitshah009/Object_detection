# coding=utf-8

import tensorflow as tf
import operator, cv2, random
from nn import resnet_fpn_backbone, is_training, roi_align, dense, wd_cost, crop_and_resize
from models import resizeImage
import numpy as np
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
			model = DCR_Resnet(config,gpuid=gpuid)

	return model


def compute_AP(lists):
	lists.sort(key=operator.itemgetter(0), reverse=True)
	rels = 0
	rank = 0
	score = 0.0
	for one in lists:
		rank+=1
		if(int(one[1]) == 1):
			rels+=1
			score+=rels/float(rank)
	if(rels != 0):
		score/=float(rels)
	return score

class DCR_Resnet():
	def __init__(self, config, gpuid=0):
		self.gpuid = gpuid
		# for batch_norm
		global is_training
		is_training = config.is_train # change this before building model

		self.config = config

		self.num_class = config.num_class

		self.global_step = tf.get_variable("global_step",shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False)

		# current model get one image at a time
		# crop multiple boxes to extract
		self.image = tf.placeholder(tf.float32,[None, None, 3],name="image")
		# used for dropout switch
		self.is_train = tf.placeholder("bool",[],name='is_train')

		# batch_size in training, testing is variable size
		# boxes should be in x1, y1, x2, y2 format
		self.boxes = tf.placeholder(tf.float32,[None, 4],name="boxes")

		self.gt_labels = tf.placeholder(tf.int64,[None,],name="gt_labels")
		self.sample_weights = tf.placeholder(tf.float32,[None,],name="sample_weights") # weight for each sample

		# the following will be added in the build_forward and loss
		self.logits = None
		self.yp = None
		self.loss = None

		self.build_preprocess()
		self.build_forward()
		if config.is_train:
			self.build_loss()

	def build_preprocess(self):
		config = self.config
		image = self.image

		bgr = True  # cv2 load image is bgr
		p_image = tf.expand_dims(image,0)  # [1,H,W,C]

		with tf.name_scope("image_preprocess"):  # tf.device("/cpu:0"):
			if p_image.dtype.base_dtype != tf.float32:
				p_image = tf.cast(p_image,tf.float32)

			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]

			p_image = p_image*(1.0/255)

			if bgr:
				mean = mean[::-1]
				std = std[::-1]
			image_mean = tf.constant(mean, dtype=tf.float32)
			image_std = tf.constant(std, dtype=tf.float32)
			p_image = (p_image - image_mean) / image_std
			p_image = tf.transpose(p_image,[0, 3, 1, 2])

		self.p_image = p_image

	def build_forward(self):
		config = self.config
		image = self.p_image # [1, 3, H, W]
		image_shape2d = tf.shape(image)[2:]

		# [N, 3, box_size, box_size]
		boxes = self.boxes
		boxes = tf.stop_gradient(boxes)
		box_images = crop_and_resize(image, boxes, tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32), config.box_size)
		# box_images = roi_align(image, boxes, config.box_size)

		# [N, C, FS, FS]
		c2, c3, c4, c5 = resnet_fpn_backbone(box_images, config.resnet_num_block, use_gn=False, resolution_requirement=32.0, use_dilations=False, use_deformable=False, tf_pad_reverse=True, freeze=config.freeze, use_basic_block=config.use_basic_block, use_se=config.use_se)
		# box_size must be divided by 32, like 224,
		c5 = tf.reshape(c5, [-1, 2048, config.box_size/32, config.box_size/32])

		# fully-connected for classification
		with tf.variable_scope("dcr_classification"):
			#dim = config.dcr_fc_dim
			initializer = tf.variance_scaling_initializer()

			hidden = c5
			#hidden = dense(c5, dim, W_init=initializer, activation=tf.nn.relu, scope="fc")
			#hidden = dense(hidden, dim, W_init=initializer, activation=tf.nn.relu, scope="fc7")

			classification = dense(hidden, config.num_class, W_init=tf.random_normal_initializer(stddev=0.01), scope="class") # [K, num_class]

		self.logits = classification
		self.yp = tf.nn.softmax(classification)

	def build_loss(self):
		config = self.config

		losses = []
		with tf.variable_scope("dcr_losses"):
			gt_labels = self.gt_labels # [num_boxes]
			logits = self.logits # [num_boxes, num_classes]

			if config.use_weighted_loss:
				label_loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels, logits=logits, weights=self.sample_weights, reduction=tf.losses.Reduction.NONE)
			else:
				label_loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels, logits=logits, reduction=tf.losses.Reduction.NONE)

			label_loss = tf.reduce_mean(label_loss, name="label_loss")

			self.box_label_loss = label_loss
			losses.append(label_loss)

			self.wd = None
			if config.wd is not None:
				wd = wd_cost('.*/W', config.wd, scope="wd_cost")
				self.wd = wd
				losses.append(wd)

		self.loss = tf.add_n(losses, 'total_loss')

	def get_feed_dict(self, batch, is_train=False):
		config = self.config

		N = len(batch.data['imgs'])

		assert N==1 # one image per gpu for now

		feed_dict = {}

		image = batch.data['imgs'][0]
		if batch.data.has_key("imgdata"):
			image = batch.data['imgdata'][0]
		else:
			image = cv2.imread(image, cv2.IMREAD_COLOR)

		assert image is not None,image
		image = image.astype("float32")

		h,w = image.shape[:2] # original width/height

		# resize image, boxes
		short_edge_size = config.short_edge_size

		if batch.data.has_key("resized_image"):
			resized_image = batch.data['resized_image'][0]
		else:
			resized_image = resizeImage(image, short_edge_size, config.max_size)
		newh,neww = resized_image.shape[:2]

		if is_train:
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box
			labels = anno['labels']
			assert len(labels) == len(o_boxes)

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:, [0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2

			# random horizontal flip
			if config.flip_image:
				prob = 0.5
				rand = random.random()
				if rand > prob:
					resized_image = cv2.flip(resized_image, 1) # 1 for horizontal
					#boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
					boxes[:,0] = neww - boxes[:,0]
					boxes[:,0,:] = boxes[:,0,::-1]# (x_min will be x_max after flip)		

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:, [0,2,1,3]] #(x1,y1,x2,y2)

			assert len(boxes) > 0

			# for training, random select mini-batch of boxes
			# 1. random replicate boxes if not enough for a mini-batch
			if len(boxes) < config.train_box_batch_size:
				need = config.train_box_batch_size - len(boxes)
				replicate_indexes = np.random.choice(len(boxes), size=need, replace=True)
				full_indexes = np.concatenate([np.arange(len(boxes)), replicate_indexes])
				boxes = boxes[full_indexes, :]
				labels = labels[full_indexes]

			selected = np.random.choice(len(boxes), size=config.train_box_batch_size, replace=False)
			boxes = boxes[selected, :]
			labels = labels[selected]

			feed_dict[self.boxes] = boxes
			feed_dict[self.gt_labels] = labels

			# different weight for each sample in the mini-batch
			if config.use_weighted_loss:
				sample_weights = np.zeros((len(boxes), ), dtype="float")
				for i in xrange(len(boxes)):
					gt_class = labels[i]
					weight = config.class_weights[gt_class]
					sample_weights[i] = weight
				feed_dict[self.sample_weights] = sample_weights
		else:
			# scale the boxes only
			anno = batch.data['gt'][0] # 'boxes' -> [K,4], 'labels' -> [K]
			o_boxes = anno['boxes'] # now the box is in [x1,y1,x2,y2] format, not coco box

			# boxes # (x1,y1,x2,y2)
			boxes = o_boxes[:, [0,2,1,3]] #(x1,x2,y1,y2)
			boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
			boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
			boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2

			boxes = boxes.reshape((-1,4))
			boxes = boxes[:, [0,2,1,3]] #(x1,y1,x2,y2)

			assert len(boxes) > 0

			feed_dict[self.boxes] = boxes


		feed_dict[self.image] = resized_image

		feed_dict[self.is_train] = is_train

		return feed_dict

	def get_feed_dict_forward(self, data):
		config = self.config

		feed_dict = {}

		image = data['image']
		boxes = data['boxes']

		feed_dict[self.boxes] = boxes

		feed_dict[self.image] = image

		feed_dict[self.is_train] = False

		return feed_dict




		