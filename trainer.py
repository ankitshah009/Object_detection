# coding=utf-8
# trainer class, given the model (model has the function to get_loss())



import tensorflow as tf
import sys

def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
		over the devices. The inner list ranges over the different variables.
	Returns:
			List of pairs of (gradient, variable) where the gradient has been averaged
			across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):

		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = [g for g, _ in grad_and_vars]
		grad = tf.reduce_mean(grads, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

class Trainer():
	def __init__(self,models,config):
		self.config = config
		self.models = models 
		self.global_step = models[0].global_step # 
		

		learning_rate = config.init_lr

		if config.use_lr_decay:
			decay_steps = int(config.train_num_examples / config.im_batch_size * config.num_epoch_per_decay)
			print "LR_decay: every %s steps"%decay_steps
			learning_rate = tf.train.exponential_decay(
			 	config.init_lr,
				self.global_step,
				decay_steps,
				config.learning_rate_decay,
				staircase=True
			)
		if config.optimizer == 'adadelta':
			self.opt = tf.train.AdadeltaOptimizer(learning_rate)
		elif config.optimizer == "adam":
			self.opt = tf.train.AdamOptimizer(learning_rate)
		elif config.optimizer == "sgd":
			self.opt = tf.train.GradientDescentOptimizer(learning_rate)
		elif config.optimizer == "momentum":
			self.opt = tf.train.MomentumOptimizer(learning_rate,momentum=config.momentum)
		else:
			print "optimizer not implemented"
			sys.exit()



		# for debug
		self.rpn_label_loss = models[0].rpn_label_loss
		self.rpn_box_loss = models[0].rpn_box_loss
		self.fastrcnn_label_loss = models[0].fastrcnn_label_loss
		self.fastrcnn_box_loss = models[0].fastrcnn_box_loss

		#self.loss = model.loss # get the loss funcion
		#self.grads = self.opt.compute_gradients(self.loss)

		self.losses = []
		self.grads = []
		for model in self.models:
			self.losses.append(model.loss)
			grad = self.opt.compute_gradients(model.loss)

			grad = [(g,var) for g, var in grad if g is not None] # we freeze resnet, so there will be none gradient

			# whehter to clip gradient
			if config.clip_gradient_norm is not None:
				grad = [(tf.clip_by_value(g, -1*config.clip_gradient_norm, config.clip_gradient_norm), var) for g, var in grad]
			self.grads.append(grad)
		
		# apply gradient on the controlling device
		with tf.device(config.controller):
			avg_loss = tf.reduce_mean(self.losses)
			avg_grads = average_gradients(self.grads)

			self.train_op = self.opt.apply_gradients(avg_grads,global_step=self.global_step)
			self.loss = avg_loss

		

	def step(self,sess,batch,get_summary=False): 
		assert isinstance(sess,tf.Session)
		config = self.config

		# idxs is a tuple (23,123,33..) index for sample
		batchIdx,batch_datas = batch
		#assert len(batch_datas) == len(self.models) # there may be less data in the end
		
		feed_dict = {}
	
		for batch_data,model in zip(batch_datas,self.models):
			feed_dict.update(model.get_feed_dict(batch_data,is_train=True))
		
		if config.add_act:
			out = [self.loss,self.rpn_label_loss, self.rpn_box_loss, self.fastrcnn_label_loss, self.fastrcnn_box_loss,self.train_op]
			out = self.models[0].act_losses + out
			things = sess.run(out,feed_dict=feed_dict)
			act_losses = things[:len(self.models[0].act_losses)]
			loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op = things[len(self.models[0].act_losses):]
		else:
			loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op = sess.run([self.loss,self.rpn_label_loss, self.rpn_box_loss, self.fastrcnn_label_loss, self.fastrcnn_box_loss,self.train_op],feed_dict=feed_dict)
			act_losses = None
		
		return loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op, act_losses

	


