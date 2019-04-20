# coding=utf-8
# trainer class, given the model (model has the function to get_loss())



import tensorflow as tf
import sys
from models import assign_to_device


# tower_grads are produced from tf.gradients, different from compute_gradients
def average_gradients(tower_grads,sum_grads=False):
	"""Calculate the average/summed gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
		over the devices. The inner list ranges over the different variables.
	Returns:
			List of pairs of (gradient, variable) where the gradient has been averaged
			across all towers.
	"""
	average_grads = []
	nr_tower = len(tower_grads)
	for grad_and_vars in zip(*tower_grads):

		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		#grads = [g for g, _ in grad_and_vars]

		grads = grad_and_vars
		if sum_grads:
			#grad = tf.reduce_sum(grads, 0)
			grad = tf.add_n(grads)
		else:
			grad = tf.multiply(tf.add_n(grads), 1.0 / nr_tower)
			#grad = tf.reduce_mean(grads, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		#v = grad_and_vars[0][1]
		average_grads.append(grad)
	return average_grads

class Trainer():
	def __init__(self,models,config):
		self.config = config
		self.models = models 
		self.global_step = models[0].global_step # 
		
		learning_rate = config.init_lr

		if config.use_lr_decay:					
			if config.use_cosine_and_warm_up:
				warm_up_start = config.init_lr * 0.33
				# linear increasing from 0.33*lr to lr in warm_up_steps
				warm_up_lr = tf.train.polynomial_decay(
					warm_up_start,
					self.global_step,
					config.warm_up_steps,
					config.init_lr,
					power=1.0, 
				)

				max_steps = int(config.train_num_examples / config.im_batch_size * config.num_epochs)
				cosine_lr = tf.train.cosine_decay(
				 	config.init_lr,
					self.global_step - config.warm_up_steps,
					max_steps - config.warm_up_steps,
					alpha=0.0
				)

				boundaries = [config.warm_up_steps] # before reaching warm_up steps, use the warm up learning rate.
				values = [warm_up_lr, cosine_lr]
				learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
				print "learning rate warm up lr from %s to %s in %s steps, then cosine learning rate decay till %s steps" % (warm_up_start, config.init_lr, config.warm_up_steps, max_steps)
			else:
				decay_steps = int(config.train_num_examples / config.im_batch_size * config.num_epoch_per_decay)
				learning_rate = tf.train.exponential_decay(
				 	config.init_lr,
					self.global_step,
					decay_steps,
					config.learning_rate_decay,
					staircase=True
				)
				print "learning rate exponential_decay: every %s steps then lr*%s" % (decay_steps, config.learning_rate_decay)

			self.learning_rate = learning_rate
		else:
			self.learning_rate = None

		last_layer_lr_mul = 10.0
		if config.optimizer == 'adadelta':
			self.opt_cnn = tf.train.AdadeltaOptimizer(learning_rate)
			self.opt_fc = tf.train.AdadeltaOptimizer(learning_rate*last_layer_lr_mul)
		elif config.optimizer == "adam":
			self.opt_cnn = tf.train.AdamOptimizer(learning_rate)
			self.opt_fc = tf.train.AdamOptimizer(learning_rate*last_layer_lr_mul)
		elif config.optimizer == "sgd":
			self.opt_cnn = tf.train.GradientDescentOptimizer(learning_rate)
			self.opt_fc = tf.train.GradientDescentOptimizer(learning_rate*last_layer_lr_mul)
		elif config.optimizer == "momentum":
			self.opt_cnn = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
			self.opt_fc = tf.train.MomentumOptimizer(learning_rate*last_layer_lr_mul, momentum=config.momentum)
		else:
			print "optimizer not implemented"
			sys.exit()

		self.box_label_losses = [model.box_label_loss for model in models]
		if config.wd is not None:
			self.wd = [model.wd for model in models]

		self.losses = []
		self.grads_cnn = []
		self.grads_fc = []
		for model in self.models:
			gpuid = model.gpuid
			# compute gradients on each gpu devices
			with tf.device(assign_to_device("/gpu:%s"%(gpuid), config.controller)):

				self.losses.append(model.loss)

				var_cnn = [var for var in tf.trainable_variables() if not var.name.startswith("dcr_classification")]
				var_fc = [var for var in tf.trainable_variables() if var.name.startswith("dcr_classification")]
				grads = tf.gradients(model.loss, var_cnn + var_fc)
				

				not_valid_idxs = [i for i in xrange(len(grads)) if grads[i] is None]
				grads = [grads[i] for i in xrange(len(grads)) if i not in not_valid_idxs] # we freeze resnet, so there will be none gradient
				var_cnn = [var_cnn[i] for i in xrange(len(var_cnn)) if i not in not_valid_idxs] # we assume fc variable all are not freezed

				# whehter to clip gradient
				if config.clip_gradient_norm is not None:
					grads = [tf.clip_by_value(g, -1*config.clip_gradient_norm, config.clip_gradient_norm) for g in grads]
				grads_cnn = grads[:len(var_cnn)]
				grads_fc = grads[len(var_cnn):]

				self.grads_cnn.append(grads_cnn)
				self.grads_fc.append(grads_fc)

				#print "valid var cnn %s, var fc %s, total valid grads %s"%(len(var_cnn), len(var_fc), len(grads))
				
		
		# apply gradient on the controlling device
		with tf.device(config.controller):
			avg_loss = tf.reduce_mean(self.losses)
			avg_grads_cnn = average_gradients(self.grads_cnn, sum_grads=True)
			avg_grads_fc = average_gradients(self.grads_fc, sum_grads=True)

			cnn_train_op = self.opt_cnn.apply_gradients(zip(avg_grads_cnn, var_cnn), global_step=self.global_step)
			fc_train_op = self.opt_fc.apply_gradients(zip(avg_grads_fc, var_fc))
			self.train_op = tf.group(cnn_train_op, fc_train_op)
			self.loss = avg_loss
		

	def step(self,sess,batch,get_summary=False): 
		assert isinstance(sess,tf.Session)
		config = self.config

		batchIdx, batch_datas = batch
		
		feed_dict = {}
	
		for batch_data, model in zip(batch_datas, self.models):
			feed_dict.update(model.get_feed_dict(batch_data, is_train=True))

		sess_input = []
		sess_input.append(self.loss)

		for i in xrange(len(self.models)):
			sess_input.append(self.box_label_losses[i])
			
			if config.wd is not None:
				sess_input.append(self.wd[i])

		sess_input.append(self.train_op)
		sess_input.append(self.learning_rate)

		outs = sess.run(sess_input, feed_dict=feed_dict)

		loss = outs[0]

		skip = 1
		box_label_losses = outs[1::skip][:len(self.models)]		
		now = 1
		wd = [-1 for m in self.models]
		if config.wd is not None:
			now+=1
			wd = outs[now::skip][:len(self.models)]

		learning_rate = outs[-1]
		return loss, wd, box_label_losses, learning_rate

	


