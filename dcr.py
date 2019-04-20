# coding=utf-8
# given the dcr annotation, do the decouple rcnn experiment.

import sys, os, argparse, json, cv2, math
import numpy as np
from tqdm import tqdm
from glob import glob
from utils import Dataset, nms_wrapper, grouper, FIFO_ME
from dcr_models import get_model, compute_AP
from dcr_trainer import Trainer
from dcr_tester import Tester, split_batch_by_box_num, split_data_by_box_num
from class_ids import targetClass2id_new_nopo
from models import resizeImage
from main import initialize, mkdir
import tensorflow as tf

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id_new_nopo[one]:one for one in targetClass2id_new_nopo}

parser = argparse.ArgumentParser()
parser.add_argument("annopath")
parser.add_argument("filelst")
parser.add_argument("framepath")
parser.add_argument("--valannopath")
parser.add_argument("--valfilelst")
parser.add_argument("--valframepath")
parser.add_argument("--train_skip", type=int, default=1, help="when load diva train set, skip how many.")
parser.add_argument("--val_skip", type=int, default=1, help="when load diva val set, skip how many.")

parser.add_argument("--outbasepath", default=None)
parser.add_argument("--modelname",type=str,default=None)

parser.add_argument("--mode", default="train|forward")

parser.add_argument("--num_class", default=15, type=int)

parser.add_argument("--use_mul", action="store_true", help="for forward mode, if set will use new prob*ori_prob instead of new prob only")

# the image size
parser.add_argument("--max_size",type=int,default=1920,help="num roi per image for RPN and fastRCNN training")
parser.add_argument("--short_edge_size",type=int,default=1080,help="num roi per image for RPN and fastRCNN training")
# the box resize size
parser.add_argument("--box_size", type=int, default=224)

# model details
parser.add_argument("--resnet50", action="store_true")
parser.add_argument("--resnet152", action="store_true")
parser.add_argument("--resnet34", action="store_true")
parser.add_argument("--resnet18", action="store_true")
parser.add_argument("--use_se", action="store_true", help="use squeeze and excitation")

# general hyperparam
parser.add_argument("--use_all_mem",action="store_true")
parser.add_argument('--im_batch_size',type=int,default=1)
parser.add_argument("--runId",type=int,default=1)
parser.add_argument("--gpu",default=1,type=int,help="number of gpu")
parser.add_argument("--gpuid_start",default=0,type=int,help="start of gpu id")
parser.add_argument("--model_per_gpu",default=1,type=int,help="it will be set as a /task:k in device")
parser.add_argument("--controller",default="/cpu:0",help="controller for multigpu training")
parser.add_argument("--load",action="store_true")
parser.add_argument("--load_best",action="store_true")
parser.add_argument("--load_from",type=str,default=None)
parser.add_argument("--ignore_gn_vars",action="store_true", help="add gn to previous model, will ignore loading the gn var first")
parser.add_argument("--ignore_vars",type=str,default=None,help="variables to ignore, multiple seperate by : like: logits/W:logits/b, this var only need to be var name's sub string to ignore")
parser.add_argument("--skip_first_eval",action="store_true")
parser.add_argument("--force_first_eval",action="store_true")
parser.add_argument("--best_first",type=float,default=None)

# training hyperparam
parser.add_argument("--train_box_batch_size", type=int, default=64)
parser.add_argument("--test_box_batch_size", type=int, default=250)
parser.add_argument("--flip_image",action="store_true",help="for training, whether to random horizontal flipping for input image, maybe not for surveillance video")
parser.add_argument("--freeze",type=int,default=0,help="freeze backbone resnet until group 0|2")
parser.add_argument("--show_loss_period", type=int, default=1000)
parser.add_argument("--loss_me_step", type=int, default=100, help="moving average queue size")
parser.add_argument("--wd",default=None,type=float)# 0.0001
parser.add_argument("--init_lr",default=0.001,type=float,help=("start learning rate"))
parser.add_argument("--use_lr_decay",action="store_true")
parser.add_argument("--learning_rate_decay",default=0.94,type=float,help=("learning rate decay"))
parser.add_argument("--num_epoch_per_decay",default=2.0,type=float,help=("how epoch after which lr decay"))
parser.add_argument("--use_cosine_and_warm_up",action="store_true")
parser.add_argument("--warm_up_steps",default=3000,type=int,help=("warm up steps not epochs"))
parser.add_argument("--optimizer",default="adam",type=str,help="optimizer: adam/adadelta")
parser.add_argument("--momentum",default=0.9,type=float)
parser.add_argument("--clip_gradient_norm",default=None,type=float,help=("norm to clip gradient to")) 

parser.add_argument("--num_epochs",type=int,default=12)
parser.add_argument("--save_period",type=int,default=5000,help="num steps to save model and eval")

# tricks for unbalance data
parser.add_argument("--use_weighted_loss", action="store_true", help="use pre-defined weight")
parser.add_argument("--oversample_min_obj_img", action="store_true")
parser.add_argument("--oversample_x", type=int, default=1, help="x + 1 times oversample")

# faster rcnn related
parser.add_argument("--result_score_thres",default=0.0001,type=float)
parser.add_argument("--result_per_im",default=100,type=int)
parser.add_argument("--fastrcnn_nms_iou_thres",type=float,default=0.5)

def read_data(config, filelst, annopath, framepath, is_train=False):

	imgs = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(filelst,"r").readlines()]

	data = {"imgs":[], "gt":[]}

	if (config.train_skip > 1) and is_train:
		imgs.sort()
		ori_num = len(imgs)
		imgs = imgs[::config.train_skip]
		print "skipping [::%s], got %s/%s"%(config.train_skip, len(imgs), ori_num)
	if (config.val_skip > 1) and not is_train:
		imgs.sort()
		ori_num = len(imgs)
		imgs = imgs[::config.val_skip]
		print "skipping [::%s], got %s/%s"%(config.val_skip, len(imgs), ori_num)

	min_objects = ["Prop", "Push_Pulled_Object", "Bike"]
	min_objects_ids = [targetClass2id[one] for one in min_objects]
	for img in tqdm(imgs, ascii=True):
		anno = os.path.join(annopath, "%s.npz"%img)
		if not os.path.exists(anno):
			continue

		anno = np.load(anno)

		# need class-agnostic models
		boxes = anno['frcnn_boxes'] # [K, 4]
		labels = anno['det_labels'] # [K] # int, 0 is background
		


		assert len(boxes) == len(labels)
		assert max(labels) < config.num_class

		videoname = img.strip().split("_F_")[0]
		data['imgs'].append(os.path.join(framepath, videoname, "%s.jpg"%img))

		# oversample some images containing minority objects
		if is_train and config.oversample_min_obj_img:
			has_min_object = np.any(np.isin(min_objects_ids, labels))
			if has_min_object:
				for i in xrange(config.oversample_x):
					data['imgs'].append(os.path.join(framepath, videoname, "%s.jpg"%img))
					data['gt'].append({
						"boxes": boxes, 
						"labels": labels,
					})

		if is_train:
			data['gt'].append({
				"boxes": boxes, 
				"labels": labels,
			})
		else:
			data['gt'].append({
				"boxes": boxes, 
				"labels": labels,
				"frcnn_probs": anno['frcnn_probs'], # [C, K]
			})
	print "loaded %s/%s data" % (len(data['imgs']), len(imgs))

	return Dataset(data, add_gt=True)

def train(config):
	eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object", "Bike"]
	eval_target = {one:1 for one in eval_target}

	# for weighted loss
	"""
	"BG":0,
	"Vehicle":1,
	"Person":2,
	"Parking_Meter":3,
	"Tree":4,
	"Skateboard":5,
	"Prop_Overshoulder":6,
	"Construction_Barrier":7,
	"Door":8,
	"Dumpster":9,
	"Push_Pulled_Object":10,
	"Construction_Vehicle":11,
	"Prop":12,
	"Bike":13,
	"Animal":14,
	"""
	# for weighted loss if used
	config.class_weights = {i:1.0 for i in xrange(config.num_class)}
	config.class_weights[10] = 2.0
	config.class_weights[12] = 2.0
	config.class_weights[13] = 2.0

	train_data = read_data(config, config.filelst, config.annopath, config.framepath, is_train=True)
	val_data = read_data(config, config.valfilelst, config.valannopath, config.valframepath, is_train=False)
	config.train_num_examples = train_data.num_examples

	# the total step (iteration) the model will run
	num_steps = int(math.ceil(train_data.num_examples/float(config.im_batch_size)))*config.num_epochs
	num_val_steps = int(math.ceil(val_data.num_examples/float(config.im_batch_size)))*1

	models = []
	gpuids = range(config.gpuid_start, config.gpuid_start+config.gpu)
	gpuids = gpuids * config.model_per_gpu 
	# example, model_per_gpu=2, gpu=2, gpuid_start=0
	gpuids.sort()
	taskids = range(config.model_per_gpu) * config.gpu # [0,1,0,1]

	for i,j in zip(gpuids,taskids):
		models.append(get_model(config, gpuid=i, task=j, controller=config.controller))

	config.is_train=False
	models_eval = []
	for i,j in zip(gpuids,taskids):
		models_eval.append(get_model(config,gpuid=i,task=j,controller=config.controller))
	config.is_train=True

	trainer = Trainer(models,config)
	tester = Tester(models_eval,config) # need final box and stuff?

	saver = tf.train.Saver(max_to_keep=5) # how many model to keep
	bestsaver = tf.train.Saver(max_to_keep=5) # just for saving the best model

	# start training!
	# allow_soft_placement :  tf will auto select other device if the tf.device(*) not available

	tfconfig = tf.ConfigProto(allow_soft_placement=True)#,log_device_placement=True)
	if not config.use_all_mem:
		tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all

	tfconfig.gpu_options.visible_device_list = "%s"%(",".join(["%s"%i for i in range(config.gpuid_start, config.gpuid_start+config.gpu)])) # so only this gpu will be used
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=tfconfig) as sess:
		initialize(load=config.load, load_best=config.load_best, config=config, sess=sess)

		isStart = True

		best = (-1.0, 1, "AP_mul")
		loss_me, box_label_loss_me, wd_me, lr_me = [FIFO_ME(config.loss_me_step) for i in xrange(4)]
		for batch in tqdm(train_data.get_batches(config.im_batch_size, num_batches=num_steps),total=num_steps,ascii=True,smoothing=1):
			global_step = sess.run(models[0].global_step) + 1 # start from 0 or the previous step

			validation_performance = None
			if (global_step % config.save_period == 0) or (config.load and isStart and ((config.ignore_vars is None) or config.force_first_eval)): # time to save model
				tqdm.write("step:%s/%s (epoch:%.3f)"%(global_step,num_steps,(config.num_epochs*global_step/float(num_steps))))
				tqdm.write("\tsaving model %s..."%global_step)
				saver.save(sess,os.path.join(config.save_dir,"model"),global_step=global_step)
				tqdm.write("\tdone")
				if config.skip_first_eval and isStart:
					tqdm.write("skipped first eval...")
					validation_performance = config.best_first
					this_val_best_type = "null"
				else:
					e = {one:[] for one in eval_target.keys()}
					e_mul = {one:[] for one in eval_target.keys()} # this will be produced by drc_prob * frcnn_prob
					for val_batch_ in tqdm(val_data.get_batches(config.im_batch_size, num_batches=num_val_steps, shuffle=False), total=num_val_steps, ascii=True, smoothing=1):
						batch_idx, val_batches = val_batch_
						this_batch_num = len(val_batches)
						# multiple image at a time for parallel inferencing with multiple gpu
						imgids = []
						for val_batch in val_batches:
							# load the image here and resize
							image = cv2.imread(val_batch.data['imgs'][0], cv2.IMREAD_COLOR)
							imgid = os.path.splitext(os.path.basename(val_batch.data['imgs'][0]))[0]
							imgids.append(imgid)
							assert image is not None, image
							image = image.astype("float32")
							val_batch.data['imgdata'] = [image]

							resized_image = resizeImage(image, config.short_edge_size, config.max_size)

							# rememember the scale and original image
							ori_shape = image.shape[:2]
							#print image.shape, resized_image.shape
							# average H/h and W/w ?

							val_batch.data['resized_image'] = [resized_image]

						# since the val_batch['boxes'] could be (1000, 4), we need to break them down
						split_val_batches = split_batch_by_box_num(val_batch_, config.test_box_batch_size)

						ori_box_nums = [b.data['gt'][0]['boxes'].shape[0] for b in val_batches]

						outputs = [[] for _ in xrange(this_batch_num)]
						for split_val_batch in split_val_batches:
							this_outputs = tester.step(sess, split_val_batch)
							for i, this_output in enumerate(this_outputs):
								outputs[i].append(this_output[0]) # [K, num_class]

						# re-asssemble the boxes
						for i in xrange(len(outputs)):
							outputs[i] = np.concatenate(outputs[i], axis=0)[:ori_box_nums[i], :]

						# post process this batch, also remember the ground truth
						for i in xrange(this_batch_num): # num gpu
							imgid = imgids[i]

							box_logit = outputs[i] # [K, num_class]
							
							val_batch = val_batches[i]

							anno = val_batch.data['gt'][0] # one val_batch is single image

							assert len(anno['boxes']) == len(anno['labels']) == len(box_logit)

							for eval_class in e:
								classIdx = targetClass2id[eval_class]

								# (K scores, K 1/0 labels)
								bin_labels = anno['labels'] == classIdx
								this_logits = box_logit[:, classIdx] # [K]
								# frcnn is [num_class-1, K]
								this_logits_mul = this_logits * anno['frcnn_probs'][classIdx-1, :]

								e[eval_class].extend(zip(this_logits, bin_labels))
								e_mul[eval_class].extend(zip(this_logits_mul, bin_labels))
					aps = []
					aps_mul = []
					for eval_class in e:
						AP = compute_AP(e[eval_class])
						aps.append((eval_class, AP))
						AP_mul = compute_AP(e_mul[eval_class])
						aps_mul.append((eval_class, AP_mul))
					average_ap = np.mean([ap for _, ap in aps])
					average_ap_mul = np.mean([ap for _, ap in aps_mul])

					validation_performance = max([average_ap_mul, average_ap])
					this_val_best_type = "AP_mul" if average_ap_mul >= average_ap else "AP"

					details = "|".join(["%s:%.5f"%(classname, ap) for classname, ap in aps])
					details_mul = "|".join(["%s:%.5f"%(classname, ap) for classname, ap in aps_mul])

					tqdm.write("\tval in %s at step %s, mean AP:%.5f, details: %s ---- mean AP_mul is %.5f, details: %s. ---- using AP_mul, previous best at %s is %.5f, type: %s"%(num_val_steps, global_step, average_ap, details, average_ap_mul, details_mul, best[1], best[0], best[2]))

				if validation_performance > best[0]:
					tqdm.write("\tsaving best model %s..." % global_step)
					bestsaver.save(sess,os.path.join(config.save_dir_best, "model"), global_step=global_step)
					tqdm.write("\tdone")
					best = (validation_performance, global_step, this_val_best_type)

				isStart = False
			
			# skip if the batch is not complete, usually the last few ones
			# lazy as fuck
			if len(batch[1]) != config.gpu:
				continue

			try:
				loss, wds, box_label_losses, lr = trainer.step(sess,batch)
			except Exception as e:
				print e
				bs = batch[1]
				print "trainer error, batch files:%s"%([b.data['imgs'] for b in bs])
				sys.exit()
			

			if math.isnan(loss):
				tqdm.write("warning, nan loss: loss:%s, box_label_loss:%s"%(loss, box_label_losses))
				print "batch:%s"%([b.data['imgs'] for b in batch[1]])
				sys.exit()

			# use moving average to compute loss

			loss_me.put(loss)
			lr_me.put(lr)
			for wd, box_label_loss in zip(wds, box_label_losses):
				wd_me.put(wd)
				box_label_loss_me.put(box_label_loss)

			if global_step % config.show_loss_period == 0:
				tqdm.write("step %s, moving average: learning_rate %.6f, loss %.6f, weight decay loss %.6f, box_label_loss %.6f" % (global_step, lr_me.me(), loss_me.me(), wd_me.me(), box_label_loss_me.me()))

def forward(config):
	# the annopath is the box output from fastrcnn model
	# given the filelst, framepath, annopath, we get new classification score for each box, then do nms, then get the final json output
	all_filenames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(config.filelst, "r").readlines()]
	print "total image to test %s"%len(all_filenames)

	if not os.path.exists(config.outbasepath):
		os.makedirs(config.outbasepath)

	models = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models.append(get_model(config, i, controller=config.controller))
	tester = Tester(models,config) # need final box and stuff?

	model_box_logits = [model.yp for model in models]

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	if not config.use_all_mem:
		tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	
	tfconfig.gpu_options.visible_device_list = "%s"%(",".join(["%s"%i for i in range(config.gpuid_start, config.gpuid_start+config.gpu)])) # so only this gpu will be used

	with tf.Session(config=tfconfig) as sess:

		initialize(load=True, load_best=config.load_best, config=config, sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1

		for filenames in tqdm(grouper(all_filenames, config.im_batch_size),ascii=True):
			filenames = [filename for filename in filenames if filename is not None]
			this_batch_num = len(filenames)

			if this_batch_num != config.im_batch_size:
				need = config.im_batch_size - this_batch_num
				filenames.extend(all_filenames[:need])
			
			ori_probs = []
			ori_frcnn_boxes = []
			data = []
			ori_box_nums = []
			for i, filename in enumerate(filenames):
				videoname = filename.split("_F_")[0]
				image = os.path.join(config.framepath, videoname, "%s.jpg"%filename)
				box_npz = os.path.join(config.annopath, "%s.npz"%filename)

				box_data = dict(np.load(box_npz))
				im = cv2.imread(image, cv2.IMREAD_COLOR)
				
				ori_shape = im.shape[:2]

				resized_image = resizeImage(im, config.short_edge_size, config.max_size)

				# [K, 4] 
				boxes = box_data['frcnn_boxes']

				resized_boxes = resize_boxes(boxes, ori_shape[0], ori_shape[1], resized_image.shape[0], resized_image.shape[1])
				
				data.append({
					"image": resized_image, 
					"boxes": resized_boxes,
				})
				ori_box_nums.append(len(boxes))
				# [C, K]
				ori_probs.append(box_data['frcnn_probs'])
				# [C, K, 4]/ now it is [K, 4]
				ori_frcnn_boxes.append(box_data['frcnn_boxes'])

			# num_splited_batch, each is num_gpu data
			mini_datas = split_data_by_box_num(data, config.test_box_batch_size)
			outputs = [[] for _ in xrange(this_batch_num)]
			for mini_data in mini_datas:

				#sess_input = []
				#for _, box_logits in zip(range(len(filenames)), model_box_logits):
				#	sess_input+=[box_logits]
				#feed_dict = {}
				#for i in xrange(this_batch_num):
				#	feed_dict.update(models[i].get_feed_dict_forward(mini_data[i]))
				#this_outputs = sess.run(sess_input, feed_dict=feed_dict)
				#pn=1
				#this_outputs = [this_outputs[i*pn:(i*pn+pn)] for i in xrange(len(filenames))]
				this_outputs = tester.step(sess, mini_data)
				for i in xrange(this_batch_num):
					outputs[i].append(this_outputs[i][0]) # [num_box, num_class] 

			# re-assemble boxes
			for i in xrange(this_batch_num):
				outputs[i] = np.concatenate(outputs[i], axis=0)[:ori_box_nums[i], :]

			for i, output in enumerate(outputs):
				# [K, num_class]
				dcr_prob = output
				dcr_prob = dcr_prob[:, 1:] # [K, C]
				# [C, K]
				dcr_prob = np.transpose(dcr_prob, axes=[1, 0]) 

				C = dcr_prob.shape[0]
				# [C, K]
				final_probs = dcr_prob
				if args.use_mul:
					ori_prob = ori_probs[i]
					final_probs = ori_prob * dcr_prob
				# [C, K, 4]/[K, 4] for class agnostic
				ori_frcnn_box = ori_frcnn_boxes[i]
				if len(ori_frcnn_box.shape) == 2:
					ori_frcnn_box = np.tile(np.expand_dims(ori_frcnn_box, axis=0), [C, 1, 1])

				final_boxes, final_labels, final_probs = nms_wrapper(ori_frcnn_box, final_probs, config)

				pred = []

				for j,(box, prob, label) in enumerate(zip(final_boxes, final_probs, final_labels)):
					box[2] -= box[0]
					box[3] -= box[1] # produce x,y,w,h output

					cat_id = int(label)
					cat_name = targetid2class[cat_id]
					
					rle = None
					
					res = {
						"category_id": cat_id,
						"cat_name": cat_name, # [0-80]
						"score": float(round(prob, 4)),
						"bbox": list(map(lambda x:float(round(x,1)),box)),
						"segmentation":rle,
					}

					pred.append(res)

				# save the data
				filename = filenames[i]
				resultfile = os.path.join(config.outbasepath, "%s.json"%filename)
				with open(resultfile, "w") as f:
					json.dump(pred, f)


def resize_boxes(o_boxes, h, w, newh, neww):
	# boxes # (x1,y1,x2,y2)
	boxes = o_boxes[:, [0,2,1,3]] #(x1,x2,y1,y2)
	boxes = boxes.reshape((-1,2,2)) # (x1,x2),(y1,y2)
	boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x1,x2
	boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y1,y2

	boxes = boxes.reshape((-1,4))
	boxes = boxes[:, [0,2,1,3]] #(x1,y1,x2,y2)

	return boxes

if __name__ == "__main__":
	args = parser.parse_args()
	
	if args.use_cosine_and_warm_up:
		args.use_lr_decay = True

	if (args.outbasepath is not None) and (args.modelname is not None):
		args.outpath = os.path.join(args.outbasepath,args.modelname,str(args.runId).zfill(2))

		args.save_dir = os.path.join(args.outpath, "save")

		args.save_dir_best = os.path.join(args.outpath, "save-best")

	if args.load_best:
		args.load = True
	if args.load_from is not None:
		args.load = True

	if args.mode == "train":
		assert args.outbasepath is not None
		assert args.modelname is not None
		args.is_train = True
		mkdir(args.save_dir)
		mkdir(args.save_dir_best)
	else:
		args.is_train = False
		args.num_epochs = 1

	args.resnet_num_block = [3, 4, 23, 3] # resnet 101
	args.use_basic_block = False
	if args.resnet152:
		args.resnet_num_block = [3, 8, 36, 3]
	if args.resnet50:
		args.resnet_num_block = [3, 4, 6, 3]
	if args.resnet34:
		args.resnet_num_block = [3, 4, 6, 3]
		args.use_basic_block = True
	if args.resnet18:
		args.resnet_num_block = [2, 2, 2, 2]
		args.use_basic_block = True

	if args.mode == "train":
		train(args)
	elif args.mode == "forward":
		forward(args)