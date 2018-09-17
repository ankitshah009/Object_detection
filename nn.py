# coding=utf-8
import tensorflow as tf
import math,re
from operator import mul
import numpy as np


VERY_NEGATIVE_NUMBER = -1e30


# f(64,2) -> 32, f(32,2) -> 16 
def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))

# leaky relu, to  avoid dead relu
# f(x) = 1(x<0)   (ax)+1  (x>=0)
# in DCGAN is f(x) = max(x or ax)
def lrelu(x,leak=0.2):
	return tf.maximum(x,leak*x)

# given regex to get the parameter to do regularization
def wd_cost(regex,wd,scope):
	params = tf.trainable_variables()
	with tf.name_scope(scope):
		costs = []
		for p in params:
			para_name = p.op.name
			if re.search(regex, para_name):
				regloss = tf.multiply(tf.nn.l2_loss(p), wd, name="%s/wd"%p.op.name)
				assert regloss.dtype.is_floating, regloss
				# Some variables may not be fp32, but it should
				# be fine to assume regularization in fp32
				if regloss.dtype != tf.float32:
					regloss = tf.cast(regloss, tf.float32)
				costs.append(regloss)

		# print the names?

		if not costs:
			return tf.constant(0, dtype=tf.float32, name=scope)
		else:
			return tf.add_n(costs,name=scope)


# relation network head, for enhancing the boxes features
# boxe_feat: [K,1024]
# boxes:[K,4]
# group is the same as multi-head in the self attention paper
def relation_network(box_appearance_feat, boxes, group=16, geo_feat_dim=64, scope="RM"):
	fc_dim = group # the geo feature for each group
	with tf.variable_scope(scope):
		box_feat_dim = box_appearance_feat.get_shape().as_list()[-1] # 1024
		group_feat_dim = box_feat_dim / group
		# [K,4] -> [K,K,4]
		# given the absolute box, get the pairwise relative geometric coordinates
		box_geo_encoded = geometric_encoding(boxes,scope="geometric_encoding")
		# [K,K,4] -> [K,K,geo_feat_dim]
		box_geo_feat = dense(box_geo_encoded,geo_feat_dim,activation=tf.nn.tanh,use_bias=True,wd=None,keep_first=False, scope="geo_emb")


		# [K,K,geo_feat_dim]
		box_geo_feat = tf.transpose(box_geo_feat,perm=[2,0,1])
		box_geo_feat = tf.expand_dims(box_geo_feat,axis=0) # [1,geo_feat_dim,K,K]
		# [1,fc_dim,K,K]
		box_geo_feat_wg = conv2d(box_geo_feat,fc_dim,kernel=1,stride=1,data_format="NCHW",scope="geo_conv")
		box_geo_feat_wg = tf.squeeze(box_geo_feat_wg)
		box_geo_feat_wg = tf.transpose(box_geo_feat_wg,perm=[1,2,0])


		# -> [K,K,fc_dim]
		box_geo_feat_wg_relu = tf.nn.relu(box_geo_feat_wg)
		# [K,fc_dim,K]
		box_geo_feat_wg_relu = tf.transpose(box_geo_feat_wg_relu,perm=[0,2,1])

		# now we get the appearance stuff
		#[K,1024]
		query = dense(box_appearance_feat,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=False, scope="query_linear")
		# split head
		#[K,16,1024/16]
		query = tf.reshape(query,(-1,group,group_feat_dim))
		query = tf.transpose(query,perm=[1,0,2]) # [16,K,1024/16]

		key = dense(box_appearance_feat,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=False, scope="key_linear")
		# split head
		#[K,16,1024/16]
		key = tf.reshape(key,(-1,group,group_feat_dim))
		key = tf.transpose(key,perm=[1,0,2]) # [16,K,1024/16]

		value = box_appearance_feat

		# [16,K,K]
		logits = tf.matmul(query,key,transpose_b=True)
		logits_scaled = (1.0 / math.sqrt(float(group_feat_dim))) * logits
		logits_scaled = tf.transpose(logits_scaled,perm=[1,0,2]) # [K,16,K]

		# [K,16,K]
		weighted_logits = tf.log(tf.maximum(box_geo_feat_wg_relu,1e-6)) + logits_scaled
		weighted_softmax = tf.nn.softmax(weighted_logits)

		# need to reshape for matmul
		weighted_softmax = tf.reshape(weighted_softmax,(tf.shape(weighted_softmax)[0]*group, tf.shape(weighted_softmax)[-1]))

		#[K*16,K] * [K,1024] -> [K*16,1024]
		output = tf.matmul(weighted_softmax,value)

		#[K,16,1024]
		output = tf.reshape(output,(-1,group,box_feat_dim))

		#[K,1024]
		output = dense(output,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=True, scope="output_linear")

		return output

# [K,4] -> [K,K,4] # get the pairwise box geometric feature
def geometric_encoding(boxes,scope="geometric_encoding"):
	with tf.variable_scope(scope):

		x1,y1,x2,y2 = tf.split(boxes, 4, axis=1)
		w = x2 - x1
		h = y2 - y1
		center_x = 0.5 * (x1+x2)
		center_y = 0.5 * (y1+y2)

		# [K,K]
		delta_x = center_x - tf.transpose(center_x)
		delta_x = delta_x / w
		delta_x = tf.log(tf.maximum(tf.abs(delta_x),1e-3))

		delta_y = center_y - tf.transpose(center_y)
		delta_y = delta_y / w
		delta_y = tf.log(tf.maximum(tf.abs(delta_y),1e-3))

		delta_w = tf.log(w / tf.transpose(w))

		delta_h = tf.log(h / tf.transpose(h))

		#[K,K,4]
		output = tf.stack([delta_x,delta_y,delta_w,delta_h],axis=2)
		
		return output



def conv2d(x,out_channel, kernel,padding="SAME",stride=1,activation=tf.identity,use_bias=True,data_format="NHWC",W_init=None,scope="conv"):
	with tf.variable_scope(scope):
		in_shape = x.get_shape().as_list()

		channel_axis = 3 if data_format == "NHWC" else 1
		in_channel = in_shape[channel_axis]

		assert in_channel is not None

		kernel_shape = [kernel,kernel]

		filter_shape = kernel_shape + [in_channel,out_channel]

		if data_format == "NHWC":
			stride = [1,stride,stride,1]
		else:
			stride = [1,1,stride,stride]

		if W_init is None:
			W_init = tf.variance_scaling_initializer(scale=2.0)
		W = tf.get_variable('W', filter_shape, initializer=W_init)

		conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)

		if use_bias:
			b_init = tf.constant_initializer()
			b = tf.get_variable('b', [out_channel], initializer=b_init)
			conv = tf.nn.bias_add(conv,b,data_format=data_format)

		ret = activation(conv,name="output")

	return ret

def deconv2d(x,out_channel, kernel,padding="SAME",stride=1,activation=tf.identity,use_bias=True,data_format="NHWC",W_init=None,scope="deconv"):

	with tf.variable_scope(scope):
		in_shape = x.get_shape().as_list()

		channel_axis = 3 if data_format == "NHWC" else 1
		in_channel = in_shape[channel_axis]

		assert in_channel is not None
		kernel_shape = [kernel,kernel]


		# TODO: change the following to tf.nn.conv2d_transpose
		if W_init is None:
			W_init = tf.variance_scaling_initializer(scale=2.0)
		b_init = tf.constant_initializer()

		
		with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
			layer = tf.layers.Conv2DTranspose(
				out_channel, kernel_shape,
				strides=stride, padding=padding,
				data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
				activation=lambda x: activation(x, name='output'),
				use_bias=use_bias,
				kernel_initializer=W_init,
				bias_initializer=b_init,
				trainable=True)
			ret = layer.apply(x, scope=tf.get_variable_scope())
		"""
		if data_format == "NHWC":
			stride = [1,stride,stride,1]
		else:
			stride = [1,1,stride,stride]

		filter_shape = kernel_shape + [in_channel,out_channel]
		W = tf.get_variable('W', filter_shape, initializer=W_init)

		deconv = tf.nn.conv2d_transpose(x,W,output_shape=out_channel,strides=stride,padding=padding,data_format=data_format)

		if use_bias:
			b = tf.get_variable('b', [out_channel], initializer=b_init)
			deconv = tf.nn.bias_add(deconv,b,data_format=data_format)
		
		return deconv
		"""
		return ret

def rename_get_variable(mapping):
	"""
	Args:
		mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
	"""
	def custom_getter(getter, name, *args, **kwargs):
		splits = name.split('/')
		basename = splits[-1]
		if basename in mapping:
			basename = mapping[basename]
			splits[-1] = basename
			name = '/'.join(splits)
		return getter(name, *args, **kwargs)
	return custom_getter_scope(custom_getter)

from contextlib import contextmanager

@contextmanager
def custom_getter_scope(custom_getter):
	scope = tf.get_variable_scope()
	with tf.variable_scope(scope, custom_getter=custom_getter):
		yield


def resnet_bottleneck(l, ch_out, stride,tf_pad_reverse=False):
	l, shortcut = l, l
	l = conv2d(l, ch_out, 1, activation=BNReLU,scope='conv1',use_bias=False,data_format="NCHW")
	if stride == 2:
		l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0,1,tf_pad_reverse), maybe_reverse_pad(0,1,tf_pad_reverse)])
		l = conv2d(l, ch_out, 3, stride=2, activation=BNReLU, padding='VALID',scope='conv2',use_bias=False,data_format="NCHW")
	else:
		l = conv2d(l, ch_out, 3, stride=stride, activation=BNReLU,scope='conv2',use_bias=False,data_format="NCHW")
	l = conv2d(l, ch_out * 4, 1, activation=get_bn(zero_init=True),scope='conv3',use_bias=False,data_format="NCHW")
	return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False),data_format="NCHW")

def resnet_shortcut(l, n_out, stride, activation=tf.identity,data_format="NCHW"):
	n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
	if n_in != n_out:   # change dimension when channel is not the same
		if stride == 2:
			l = l[:, :, :-1, :-1]
			return conv2d(l, n_out, 1,
						  stride=stride, padding='VALID', activation=activation,use_bias=False,data_format=data_format,scope='convshortcut')
		else:
			return conv2d(l, n_out, 1,
						  stride=stride, activation=activation,use_bias=False,data_format=data_format,scope='convshortcut')
	else:
		return l

def resnet_group(l, name, block_func, features, count, stride,reuse=False,tf_pad_reverse=False):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		for i in range(0, count):
			with tf.variable_scope('block{}'.format(i)):
				l = block_func(l, features,
							   stride if i == 0 else 1,tf_pad_reverse=tf_pad_reverse)
				# end of each block need an activation
				l = tf.nn.relu(l)
	return l

def get_bn(zero_init=False):
	if zero_init:
		return lambda x, name: BatchNorm(x, gamma_init=tf.zeros_initializer(),scope="bn")
	else:
		return lambda x, name: BatchNorm(x,scope="bn")

def BNReLU(x, name=None):
	"""
	A shorthand of BatchNormalization + ReLU.
	"""
	x = BatchNorm(x,scope="bn")
	x = tf.nn.relu(x, name=name)
	return x

# TODO: replace these
#-----------------------------------

from tensorflow.contrib.framework import add_model_variable
is_training = False
def BatchNorm(x, use_local_stat=False, decay=0.9, epsilon=1e-5,
			  use_scale=True, use_bias=True,
			  gamma_init=tf.constant_initializer(1.0), data_format='NCHW',
			  internal_update=False,scope="bn"):
	global is_training
	with tf.variable_scope(scope):
		shape = x.get_shape().as_list()
		ndims = len(shape)
		assert ndims in [2, 4]
		if ndims == 2:
			data_format = 'NHWC'
		if data_format == 'NCHW':
			n_out = shape[1]
		else:
			n_out = shape[-1]  # channel
		assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
		beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias, gamma_init)

		use_local_stat = bool(use_local_stat)

		if use_local_stat:
			if ndims == 2:
				x = tf.reshape(x, [-1, 1, 1, n_out])	# fused_bn only takes 4D input
				# fused_bn has error using NCHW? (see #190)

			xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
				x, gamma, beta, epsilon=epsilon,
				is_training=True, data_format=data_format)

			if ndims == 2:
				xn = tf.squeeze(xn, [1, 2])
		else:
			if is_training: # so ugly
				#assert get_tf_version_number() >= 1.4, \
				#	"Fine tuning a BatchNorm model with fixed statistics is only " \
				#	"supported after https://github.com/tensorflow/tensorflow/pull/12580 "
				#if ctx.is_main_training_tower:  # only warn in first tower
				#	logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
				# Using moving_mean/moving_variance in training, which means we
				# loaded a pre-trained BN and only fine-tuning the affine part.
				xn, _, _ = tf.nn.fused_batch_norm(
					x, gamma, beta,
					mean=moving_mean, variance=moving_var, epsilon=epsilon,
					data_format=data_format, is_training=False)
			else:
				# non-fused op is faster for inference  # TODO test if this is still true
				if ndims == 4 and data_format == 'NCHW':
					[g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
									  for _ in [gamma, beta, moving_mean, moving_var]]
					xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
				else:
					# avoid the reshape if possible (when channel is the last dimension)
					xn = tf.nn.batch_normalization(
						x, moving_mean, moving_var, beta, gamma, epsilon)

		# maintain EMA only on one GPU is OK, even in replicated mode.
		# because training time doesn't use EMA
		#if ctx.is_main_training_tower:
		add_model_variable(moving_mean)
		add_model_variable(moving_var)
		if use_local_stat: # and ctx.is_main_training_tower:
			ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay, internal_update)
		else:
			ret = tf.identity(xn, name='output')

		return ret

def update_bn_ema(xn, batch_mean, batch_var,
				  moving_mean, moving_var, decay, internal_update):
	# TODO is there a way to use zero_debias in multi-GPU?
	update_op1 = moving_averages.assign_moving_average(
		moving_mean, batch_mean, decay, zero_debias=False,
		name='mean_ema_op')
	update_op2 = moving_averages.assign_moving_average(
		moving_var, batch_var, decay, zero_debias=False,
		name='var_ema_op')

	if internal_update:
		with tf.control_dependencies([update_op1, update_op2]):
			return tf.identity(xn, name='output')
	else:
		tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
		tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
		return xn


def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
	if use_bias:
		beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
	else:
		beta = tf.zeros([n_out], name='beta')
	if use_scale:
		gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
	else:
		gamma = tf.ones([n_out], name='gamma')
	# x * gamma + beta

	moving_mean = tf.get_variable('mean/EMA', [n_out],
								  initializer=tf.constant_initializer(), trainable=False)
	moving_var = tf.get_variable('variance/EMA', [n_out],
								 initializer=tf.constant_initializer(1.0), trainable=False)
	return beta, gamma, moving_mean, moving_var

def reshape_for_bn(param, ndims, chan, data_format):
	if ndims == 2:
		shape = [1, chan]
	else:
		shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
	return tf.reshape(param, shape)
#----------------------------------------------------


# add weight decay to the current varaible scope
def add_wd(wd,scope=None):
	if wd != 0.0:
		# for all variable in the current scope
		scope = scope or tf.get_variable_scope().name
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		with tf.name_scope("weight_decay"):
			for var in variables:
				weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="%s/wd"%var.op.name)
				tf.add_to_collection('losses', weight_decay)


# flatten a tensor
# [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
def flatten(tensor, keep): # keep how many dimension in the end, so final rank is keep + 1
	# get the shape
	fixed_shape = tensor.get_shape().as_list() #[N, JQ, di] # [N, M, JX, di] 
	start = len(fixed_shape) - keep # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
	# each num in the [] will a*b*c*d...
	# so [0] -> just N here for left
	# for [N, M, JX, di] , left is N*M
	left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
	# [N, JQ,di]
	# [N*M, JX, di] 
	out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
	# reshape
	flat = tf.reshape(tensor, out_shape)
	return flat

def reconstruct(tensor, ref, keep): # reverse the flatten function
	ref_shape = ref.get_shape().as_list()
	tensor_shape = tensor.get_shape().as_list()
	ref_stop = len(ref_shape) - keep
	tensor_start = len(tensor_shape) - keep
	pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
	keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
	# pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
	# keep_shape = tensor.get_shape().as_list()[-keep:]
	target_shape = pre_shape + keep_shape
	out = tf.reshape(tensor, target_shape)
	return out

# boxes are x1y1,x2y2
def pairwise_iou(boxes1,boxes2):
	def area(boxes): # [N,4] -> [N]
		x1,y1,x2,y2 = tf.split(boxes,4,axis=1)
		return tf.squeeze((y2-y1)*(x2-x1),[1])

	# two box list,  get intersected boxes area [N,M] 
	def pairwise_intersection(b1,b2):
		x_min1, y_min1, x_max1, y_max1 = tf.split(b1, 4, axis=1)
		x_min2, y_min2, x_max2, y_max2 = tf.split(b2, 4, axis=1)
		all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
		all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
		intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
		all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
		all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
		intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
		return intersect_heights * intersect_widths

	interarea = pairwise_intersection(boxes1,boxes2)
	areas1 = area(boxes1)#[N]
	areas2 = area(boxes2)#[M]
	unions = tf.expand_dims(areas1,1) + tf.expand_dims(areas2,0) - interarea

	# avoid zero divide?
	return tf.truediv(interarea,unions)


import pycocotools.mask as cocomask

def np_iou(A,B):
	def to_xywh(box):
		box = box.copy()
		box[:, 2] -= box[:, 0]
		box[:, 3] -= box[:, 1]
		return box

	ret = cocomask.iou(
		to_xywh(A), to_xywh(B),
		np.zeros((len(B),), dtype=np.bool))
	# can accelerate even more, if using float32
	return ret.astype('float32')


#@memorized
def get_iou_callable():
	with tf.Graph().as_default(),tf.device("/cpu:0"):
		A = tf.placeholder(tf.float32,shape=[None,4])
		B = tf.placeholder(tf.float32,shape=[None,4])
		iou = pairwise_iou(A,B)
		sess = tf.Session()
		return sess.make_callable(iou,[A,B])



# simple linear layer, without activatation # remember to add it
def dense(x,output_size,W_init=None,b_init=None,activation=tf.identity,use_bias=True,wd=None,keep_first=True, scope="dense"):
	with tf.variable_scope(scope):
		# tensorpack's fully connected keep the first dim and flatten the rest, apply W on the rest
		if keep_first:
			shape = x.get_shape().as_list()[1:]
			if None not in shape:
				flat_x = tf.reshape(x,[-1,int(np.prod(shape))])
			else:
				flat_x = tf.reshape(x,tf.stack([tf.shape(x)[0],-1]))
		else:
			# we used to apply W on the last dimention
			# since the input here is not two rank, we flat the input while keeping the last dims
			keep = 1
			#print x.get_shape().as_list()
			flat_x = flatten(x,keep) # keeping the last one dim # [N,M,JX,JQ,d] => [N*M*JX*JQ,d]
		
		if W_init is None:
			W_init = tf.variance_scaling_initializer(2.0)

		W = tf.get_variable("W",[flat_x.get_shape().as_list()[-1],output_size],initializer=W_init)

		flat_out = tf.matmul(flat_x,W)

		if use_bias:
			if b_init is None:
				b_init = tf.constant_initializer()
			b = tf.get_variable('b', [output_size], initializer=b_init)
			flat_out = tf.nn.bias_add(flat_out, b)

		flat_out = activation(flat_out)

		if wd is not None:
			add_wd(wd)

		if not keep_first:
			out = reconstruct(flat_out,x,keep)
		else:
			out = flat_out
		return out

def  maybe_reverse_pad(topleft, bottomright,reverse=False):
	if reverse:
		return [bottomright, topleft]
	else:
		return [topleft, bottomright]

def MaxPooling(x, shape, stride=None, padding='VALID', data_format='NHWC',scope="maxpooling"):
	with tf.variable_scope(scope):
		if stride is None:
			stride = shape
		ret = tf.layers.max_pooling2d(x, shape, stride, padding,
									  'channels_last' if data_format == 'NHWC' else 'channels_first')
		return tf.identity(ret, name='output')

def pretrained_resnet_conv4(image, num_blocks,tf_pad_reverse=False):
	assert len(num_blocks) == 3
	# pad 2 zeros to front of H and 3 zeros to back of H, same for W
	# so 2-3lines of zeros outside the data center
	# original H,W will be H+5,W+5
	l = tf.pad(image,[[0, 0], [0, 0], maybe_reverse_pad(2, 3,tf_pad_reverse), maybe_reverse_pad(2, 3,tf_pad_reverse)])

	l = conv2d(l, 64, 7, stride=2, activation=BNReLU, padding='VALID',scope="conv0",use_bias=False,data_format="NCHW")

	#print l.get_shape()# (1,64,?,?)
	l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1,tf_pad_reverse), maybe_reverse_pad(0, 1,tf_pad_reverse)])
	l = MaxPooling(l, shape=3, stride=2, padding='VALID',scope='pool0',data_format="NCHW")

	#print l.get_shape()# (1,64,?,?)
	l = resnet_group(l, 'group0', resnet_bottleneck, 64, num_blocks[0], stride=1,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,256,?,?)
	# TODO replace var by const to enable folding
	#l = tf.stop_gradient(l) # froze outside
	l = resnet_group(l, 'group1', resnet_bottleneck, 128, num_blocks[1], stride=2,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,512,?,?)
	l = resnet_group(l, 'group2', resnet_bottleneck, 256, num_blocks[2], stride=2,tf_pad_reverse=tf_pad_reverse)
	return l


def resnet_conv5(image,num_block,reuse=False,tf_pad_reverse=False):
	l = resnet_group(image,"group3",resnet_bottleneck,512,num_block,stride=2,reuse=reuse,tf_pad_reverse=tf_pad_reverse)
	return l

# fpn_resolution_requirement is 32 by default FPN
def resnet_fpn_backbone(image, num_blocks,resolution_requirement,tf_pad_reverse=False,finer_resolution=False):
	assert len(num_blocks) == 4
	shape2d = tf.shape(image)[2:]
	# padding to deal with odd size image?
	mult = resolution_requirement * 1.0
	new_shape2d = tf.to_int32(tf.ceil(tf.to_float(shape2d) / mult) * mult)
	pad_shape2d = new_shape2d - shape2d

	channel = image.shape[1]

	pad_base = maybe_reverse_pad(2, 3,tf_pad_reverse)
	l = tf.pad(image,[
			[0, 0], [0, 0], 
			[pad_base[0], pad_base[1] + pad_shape2d[0]], 
			[pad_base[0], pad_base[1] + pad_shape2d[1]]])
	l.set_shape([None,channel,None,None])

	# rest is the same as c4 backbone
	l = conv2d(l, 64, 7, stride=2, activation=BNReLU, padding='VALID',scope="conv0",use_bias=False,data_format="NCHW")
	c1 = l
	l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1,tf_pad_reverse), maybe_reverse_pad(0, 1,tf_pad_reverse)]) # H+1,W+1
	
	l = MaxPooling(l, shape=3, stride=2, padding='VALID',scope='pool0',data_format="NCHW")
	# here 4x down already, so smallest anchor box can on use these

	#print l.get_shape()# (1,64,?,?)
	c2 = resnet_group(l, 'group0', resnet_bottleneck, 64, num_blocks[0], stride=1,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,256,?,?)
	c3 = resnet_group(c2, 'group1', resnet_bottleneck, 128, num_blocks[1], stride=2,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,512,?,?)
	c4 = resnet_group(c3, 'group2', resnet_bottleneck, 256, num_blocks[2], stride=2,tf_pad_reverse=tf_pad_reverse)
	c5 = resnet_group(c4, "group3", resnet_bottleneck, 512, num_blocks[3], stride=2,tf_pad_reverse=tf_pad_reverse)
	## 32x downsampling up to now
	# size of c5: ceil(input/32)
	if finer_resolution:
		return c1,c2,c3,c4
	return c2,c3,c4,c5

# the FPN model
def fpn_model(c2345,num_channel,scope):

	def upsample2x(x,scope):
		with tf.name_scope(scope):
			# FPN paper uses nearest neighbour
			# a outer product with 2x2 , makes x upsampled
			unpool_mat = np.ones((2,2),dtype="float32")
			shape = (2,2)
			output_shape = x.get_shape().as_list() # [N,C,H,W]

			unpool_mat = tf.constant(unpool_mat,name="unpool_mat")

			assert unpool_mat.get_shape().as_list() == list(shape)

			# outer product with ones, so just duplicate stuff
			x = tf.expand_dims(x, -1) # NxCxHxWx1
			mat = tf.expand_dims(unpool_mat,0) # 1xSHxSW
			ret = tf.tensordot(x, mat, axes=1) # NxCxHxWxSHxSW

			ret = tf.transpose(ret, [0,1,2,4,3,5]) # NxCxHxSHxWxSW

			ret = tf.reshape(ret, tf.stack([-1, output_shape[1],tf.shape(x)[2] * shape[0], tf.shape(x)[3] * shape[1] ]))# [N,C,H*2,W*2]

			return ret



	with tf.variable_scope(scope):
		# each conv feature go through 1x1 conv, then add to 2x upsampled feature, then add 3x3 conv to get final feature
		lat_2345 = [conv2d(c, num_channel, 1, stride=1, activation=tf.identity, padding='SAME',scope="lateral_1x1_c%s"%(i+2),use_bias=True,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=1.0)) for i,c in enumerate(c2345)]

		lat_sum_5432 = []
		for idx, lat in enumerate(lat_2345[::-1]):
			if idx == 0:
				lat_sum_5432.append(lat)
			else:
				lat = lat + upsample2x(lat_sum_5432[-1],scope="upsample_lat%s"%(6 - idx))
				lat_sum_5432.append(lat)

		p2345 = [conv2d(c, num_channel, 3, stride=1, activation=tf.identity, padding='SAME',scope="posthoc_3x3_p%s"%(i+2),use_bias=True,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=1.0)) for i,c in enumerate(lat_sum_5432[::-1])]

		p6 = MaxPooling(p2345[-1], shape=1, stride=2, padding='VALID',scope='maxpool_p6',data_format="NCHW")
		return p2345+[p6]

