# coding=utf-8
# main script for training and testing mask rcnn on MSCOCO dataset
# multi gpu version

import sys,os,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info

import numpy as np

import cv2

from models import get_model,get_model_boxfeat,get_model_givenbox
from trainer import Trainer
from tester import Tester
import math,time,json,random,operator
import cPickle as pickle

import tensorflow as tf
import pycocotools.mask as cocomask

#assert tf.__version__ > "1.4.0"

from tqdm import tqdm
from models import fill_full_mask, resizeImage
from utils import evalcoco,get_op_tensor_name,match_detection,computeAP,computeAR,computeAR_2,grouper,gather_dt,gather_gt,match_dt_gt,gather_act_singles,aggregate_eval,weighted_average

from utils import Dataset,Summary

def get_args():
	global targetClass2id, targetid2class
	parser = argparse.ArgumentParser()

	parser.add_argument("datajson")
	parser.add_argument("imgpath")

	parser.add_argument("--outbasepath",type=str,default=None,help="full path will be outbasepath/modelname/runId")

	parser.add_argument("--actoutbasepath",type=str,default=None,help="for activity box forward only")

	parser.add_argument("--train_skip",type=int,default=1,help="when load diva train set, skip how many.")
	parser.add_argument("--train_skip_offset",type=int,default=0,help="when load diva train set, offset before skip")

	parser.add_argument("--val_skip",type=int,default=1,help="when load diva val set, skip how many.")
	parser.add_argument("--val_skip_offset",type=int,default=0,help="when load diva train set, offset before skip")
	parser.add_argument("--exit_after_val",action="store_true")

	parser.add_argument("--forward_skip",type=int,default=1,help="forward, skip how many.")

	parser.add_argument("--modelname",type=str,default=None)
	parser.add_argument("--num_class",type=int,default=81,help="num catagory + 1 background")


	# ------ extract fpn feature of the whole image
	parser.add_argument("--extract_feat",action="store_true")
	parser.add_argument("--feat_path",default=None)
	parser.add_argument("--just_feat",action="store_true",help="only extract full image feature no bounding box")

	# ------ do object detection and extract the fpn feature for each *final* boxes
	parser.add_argument("--get_box_feat",action="store_true")
	parser.add_argument("--box_feat_path",default=None)

	# ---different from above, only feat no object detection
	parser.add_argument("--videolst",default=None)
	parser.add_argument("--skip",action="store_true",help="skip existing npy")

	parser.add_argument("--tococo",action="store_true",help="for training in diva using coco model, map diva class1to1 to coco")
	parser.add_argument("--diva_class",action="store_true",help="the last layer is 16 (full) class output as the diva object classes")
	#parser.add_argument("--diva_class2",action="store_true",help="the last layer is 16 class output as the diva object classes")

	
	parser.add_argument("--merge_prop",action="store_true",help="use annotation that merged prop and Push_Pulled_Object and train")

	# ------------activity detection

	parser.add_argument("--act_as_obj",action="store_true",help="activity box as obj box")

	parser.add_argument("--add_act",action="store_true",help="add activitiy model")

	parser.add_argument("--fix_obj_model",action="store_true",help="fix the object detection part including rpn")
	# v1:
	parser.add_argument("--num_act_class",type=int,default=36,help="num catagory + 1 background")
	parser.add_argument("--fastrcnn_act_fg_ratio",default=0.25,type=float)
	parser.add_argument("--act_relation_nn",action="store_true",help="add relation link in activity fastrnn head")
	parser.add_argument("--act_loss_weight",default=1.0,type=float)
	# ----- activity detection version 2
	parser.add_argument("--act_v2",action="store_true")
	parser.add_argument("--act_single_topk",type=int,default=5,help="each box topk classes are output")
	parser.add_argument("--num_act_single_class",default=36,type=int)
	parser.add_argument("--num_act_pair_class",default=21,type=int)


	# ---------------------------------------------

	parser.add_argument("--debug",action="store_true",help="load fewer image for debug in training")
	parser.add_argument("--runId",type=int,default=1)

	# forward mode: imgpath is the list of images
	# will output result to outbasepath
	# forward still need a coco validation json to get the catgory names
	parser.add_argument("--mode",type=str,default="forward",help="train | test | forward | boxfeat | givenbox")
	parser.add_argument("--avg_feat",action="store_true",help="for boxfeat mode, output 7x7x2048 or just 2048 for each box")
	parser.add_argument("--boxjsonpath",default=None,help="json contain a dict for all the boxes, imageId -> boxes")
	parser.add_argument("--boxfeatpath",default=None,help="where to save the box feat path, will be a npy for each image")
	parser.add_argument("--boxclass",action="store_true",help="do box classification as well")

	parser.add_argument("--resnet152",action="store_true",help="")

	parser.add_argument("--is_fpn",action="store_true")
	parser.add_argument("--obj_v2",action="store_true")

	parser.add_argument("--fpn_frcnn_fc_head_dim",type=int,default=1024)
	parser.add_argument("--fpn_num_channel",type=int,default=256)
	parser.add_argument("--freeze",type=int,default=0,help="freeze backbone resnet until group 0|2")

	parser.add_argument("--finer_resolution",action="store_true",help="fpn use finer resolution conv")
	parser.add_argument("--is_cascade_rcnn",action="store_true",help="cascade rcnn on top of fpn")

	parser.add_argument("--add_relation_nn",action="store_true",help="add relation network feature")

	parser.add_argument("--focal_loss",action="store_true",help="use focal loss for RPN and FasterRCNN loss, instead of cross entropy")


	# for test mode on testing on the MSCOCO dataset, if not set this, will use our evaluation script
	parser.add_argument("--use_coco_eval",action="store_true")
	parser.add_argument("--coco2014_to_2017",action="store_true",help="if use the cocoval 2014 json and use val2017 filepath, need this option to get the correct file path")

	# this will alter some parameter in tf.pad in resnet bottleneck and resnet conv4,
	## In tensorpack model zoo, ResNet models with TF_PAD_MODE=False are marked with "-AlignPadding".
	# All other models under `ResNet/` in the model zoo are trained with TF_PAD_MODE=True.
	#   _C.BACKBONE.TF_PAD_MODE = False
	parser.add_argument("--new_tensorpack_model",action="store_true",help="for new tensorpack model, the fast rcnn box logit has num_class instead of num_class-1, and some padding is different")

	parser.add_argument("--trainlst",type=str,default=None,help="training frame name list,")
	parser.add_argument("--valframepath",type=str,default=None,help="path to top frame path")
	parser.add_argument("--annopath",type=str,default=None,help="path to annotation, each frame.npz")
	parser.add_argument("--valannopath",type=str,default=None,help="path to annotation, each frame.npz")
	parser.add_argument("--flip_image",action="store_true",help="for training, whether to random horizontal flipping for input image, maybe not for surveillance video")

	parser.add_argument("--add_mask",action="store_true")

	parser.add_argument("--vallst",type=str,default=None,help="validation for training")

	parser.add_argument("--load",action="store_true")
	parser.add_argument("--load_best",action="store_true")

	parser.add_argument("--skip_first_eval",action="store_true")
	parser.add_argument("--best_first",type=float,default=None)

	parser.add_argument("--no_skip_error",action="store_true")

	parser.add_argument("--show_stat",action="store_true",help="show data distribution only")

	# use for pre-trained model
	parser.add_argument("--load_from",type=str,default=None)
	parser.add_argument("--ignore_vars",type=str,default=None,help="variables to ignore, multiple seperate by : like: logits/W:logits/b, this var only need to be var name's sub string to ignore")

	parser.add_argument("--print_params",action="store_true",help="print params and then exit")
	parser.add_argument("--show_restore",action="store_true",help="load from existing model (npz), show the weight that is restored")


	# -------------------- save model for deployment
	parser.add_argument("--is_pack_model",action="store_true",default=False,help="with is_test, this will pack the model to a path instead of testing")
	parser.add_argument("--pack_model_path",type=str,default=None,help="path to save model")
	parser.add_argument("--pack_model_note",type=str,default=None,help="leave a note for this packed model for future reference")

	# ------------------------------------ model specifics
	

	
	# ----------------------------------training detail
	parser.add_argument("--use_all_mem",action="store_true")
	parser.add_argument('--im_batch_size',type=int,default=1)
	parser.add_argument("--rpn_batch_size",type=int,default=256,help="num roi per image for RPN  training")
	parser.add_argument("--frcnn_batch_size",type=int,default=512,help="num roi per image for fastRCNN training")
	
	parser.add_argument("--rpn_test_post_nms_topk",type=int,default=1000,help="test post nms, input to fast rcnn")
	# fastrcnn output NMS suppressing iou >= this thresZ
	parser.add_argument("--fastrcnn_nms_iou_thres",type=float,default=0.5)

	parser.add_argument("--max_size",type=int,default=1333,help="num roi per image for RPN and fastRCNN training")
	parser.add_argument("--short_edge_size",type=int,default=800,help="num roi per image for RPN and fastRCNN training")
	parser.add_argument("--scale_jitter",action="store_true",help="if set this, will random get int from min to max to resize image;original param will still be used in testing")
	parser.add_argument("--short_edge_size_min",type=int,default=640,help="num roi per image for RPN and fastRCNN training")
	parser.add_argument("--short_edge_size_max",type=int,default=800,help="num roi per image for RPN and fastRCNN training")

	# not used for fpn
	parser.add_argument("--small_anchor_exp",action="store_true")


	parser.add_argument("--positive_anchor_thres",default=0.7,type=float)
	parser.add_argument("--negative_anchor_thres",default=0.3,type=float)
	

	parser.add_argument("--fastrcnn_fg_ratio",default=0.25,type=float)

	parser.add_argument("--gpu",default=1,type=int,help="number of gpu")
	parser.add_argument("--gpuid_start",default=0,type=int,help="start of gpu id")
	parser.add_argument("--model_per_gpu",default=1,type=int,help="it will be set as a /task:k in device")
	parser.add_argument("--controller",default="/cpu:0",help="controller for multigpu training")
	

	#parser.add_argument("--num_step",type=int,default=360000) 
	parser.add_argument("--num_epochs",type=int,default=12)

	parser.add_argument("--save_period",type=int,default=5000,help="num steps to save model and eval")

	# drop out rate
	parser.add_argument('--keep_prob',default=1.0,type=float,help="1.0 - drop out rate;remember to set it to 1.0 in eval")

	# l2 weight decay
	parser.add_argument("--wd",default=None,type=float)# 0.0001

	parser.add_argument("--init_lr",default=0.1,type=float,help=("start learning rate"))


	parser.add_argument("--use_lr_decay",action="store_true")
	parser.add_argument("--learning_rate_decay",default=0.94,type=float,help=("learning rate decay"))
	#parser.add_argument("--learning_rate_decay_examples",default=1000000,type=int,help=("how many sample to have one decay"))
	parser.add_argument("--num_epoch_per_decay",default=2.0,type=float,help=("how epoch after which lr decay"))
	
	parser.add_argument("--optimizer",default="adam",type=str,help="optimizer: adam/adadelta")
	parser.add_argument("--momentum",default=0.9,type=float)

	# clipping, suggest 100.0
	parser.add_argument("--clip_gradient_norm",default=None,type=float,help=("norm to clip gradient to")) 


	# for debug
	parser.add_argument("--vis_pre",action="store_true",help="visualize preprocess images")
	parser.add_argument("--vis_path",default=None)

	args = parser.parse_args()

	assert args.model_per_gpu == 1, "not work yet!"
	assert args.gpu*args.model_per_gpu == args.im_batch_size # one gpu one image
	#args.controller = "/cpu:0" # parameter server

	assert int(args.diva_class) + int(args.tococo) == 1

	if args.add_act:
		assert len(targetAct2id) == args.num_act_class, (len(targetAct2id),args.num_act_class)
		assert len(targetSingleAct2id) == args.num_act_single_class, (len(targetSingleAct2id), args.num_act_single_class)
		assert len(targetPairAct2id) == args.num_act_pair_class

	targetid2class = targetid2class
	targetClass2id = targetClass2id

	
	if args.merge_prop:
		targetClass2id = targetClass2id_mergeProp
		targetid2class = {targetClass2id_mergeProp[one]:one for one in targetClass2id_mergeProp}

	if args.act_as_obj:
		# replace the obj class with actitivy class
		targetClass2id = targetAct2id
		targetid2class = {targetAct2id[one]:one for one in targetAct2id}
		

	assert len(targetClass2id) == args.num_class
	#if args.diva_class2:
	#	targetid2class = targetid2class_v2
	#	targetClass2id = targetClass2id_v2

	if not args.tococo and ((args.mode == "train") or (args.mode == "test")):
		assert args.num_class == len(targetClass2id.keys())
	args.class_names = targetClass2id.keys()

	if args.vis_pre:
		assert args.vis_path is not None
		if not os.path.exists(args.vis_path):
			os.makedirs(args.vis_path)

	if args.add_act and (args.mode == "forward"):
		assert args.actoutbasepath is not None
		mkdir(args.actoutbasepath)

	if args.outbasepath is not None:
		mkdir(args.outbasepath)

	if args.skip_first_eval:
		assert args.best_first is not None

	if (args.outbasepath is not None) and (args.modelname is not None):
		args.outpath = os.path.join(args.outbasepath,args.modelname,str(args.runId).zfill(2))

		args.save_dir = os.path.join(args.outpath, "save")

		args.save_dir_best = os.path.join(args.outpath, "save-best")

		args.write_self_sum = True
		args.self_summary_path = os.path.join(args.outpath,"train_sum.txt")
		args.stats_path = os.path.join(args.outpath,"stats.json")# path to save each validation step's performance and loss

	args.mrcnn_head_dim = 256

	args.no_obj_detect = False
	if args.mode == "videofeat":
		args.no_obj_detect = True

	if args.is_cascade_rcnn:
		assert args.is_fpn
		args.cascade_num_stage = 3
		args.cascade_ious = [0.5, 0.6, 0.7]
		
		args.cascade_bbox_reg = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]


	if args.is_fpn:
		args.anchor_strides = (4, 8, 16, 32, 64)
		if args.obj_v2:
			args.anchor_strides = (4, 8, 16, 32)

		args.fpn_resolution_requirement = float(args.anchor_strides[3]) # [3] is 32, since we build FPN with r2,3,4,5?

		
		args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * args.fpn_resolution_requirement

		#args.fpn_num_channel = 256

		#args.fpn_frcnn_fc_head_dim = 1024

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

	

	# ---- all the mask rcnn config

	args.resnet_num_block = [3,4,23,3] # resnet 101
	if args.resnet152:
		args.resnet_num_block = [3, 8, 36, 3]

	#args.short_edge_size = 800
	#args.max_size = 1333

	args.anchor_stride = 16 # has to be 16 to match the image feature total stride
	args.anchor_sizes = (32, 64, 128, 256, 512)

	if args.obj_v2:
		args.anchor_sizes = (32, 64, 128, 256)

	if args.small_anchor_exp:
		args.anchor_sizes = (16, 32, 64, 96,128, 256) # not used for fpn


	args.anchor_ratios = (0.5, 1, 2)
	

	args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
	# iou thres to determine anchor label
	#args.positive_anchor_thres = 0.7
	#args.negative_anchor_thres = 0.3

	# when getting region proposal, avoid getting too large boxes
	args.bbox_decode_clip = np.log(args.max_size / 16.0)

	# RPN training
	args.rpn_fg_ratio = 0.5
	args.rpn_batch_per_im = args.rpn_batch_size
	args.rpn_min_size = 0 # 8?

	args.rpn_proposal_nms_thres = 0.7

	args.rpn_train_pre_nms_topk = 12000 # not used in fpn
	args.rpn_train_post_nms_topk = 2000# this is used for fpn_nms_pre


	# fastrcnn
	args.fastrcnn_batch_per_im = args.frcnn_batch_size
	args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype='float32')
	#args.fastrcnn_bbox_reg_weights = np.array([20, 20, 10, 10], dtype='float32')

	args.fastrcnn_fg_thres = 0.5 # iou thres
	#args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

	# testing
	args.rpn_test_pre_nms_topk = 6000

	#args.rpn_test_post_nms_topk = 700 #1300 # 700 takes 40 hours, # OOM at 1722,28,28,1024 # 800 OOM for gpu4
	#args.fastrcnn_nms_thres = 0.5
	#args.fastrcnn_nms_iou_thres = 0.5 # 0.3?

	args.result_score_thres = 0.0001
	args.result_per_im = 100 # 400 # 100

	return args


from pycocotools.coco import COCO

def add_coco(config,datajson):
	coco = COCO(datajson)
	cat_ids = coco.getCatIds() #[80], each is 1-90
	cat_names = [c['name'] for c in coco.loadCats(cat_ids)] # [80]

	config.classId_to_cocoId = {(i+1):v for i,v in enumerate(cat_ids)}

	config.class_names = ["BG"] + cat_names

	config.class_to_classId = {c:i for i,c in enumerate(config.class_names)} # 0-80
	config.classId_to_class = {i:c for i,c in enumerate(config.class_names)}


# for using a COCO model to finetuning with DIVA data.
from class_ids import targetClass1to1, targetClass2id, targetAct2id,targetAct2id_wide,targetAct2id_tall, targetSingleAct2id, targetPairAct2id,targetClass2id_tall,targetClass2id_wide,targetClass2id_wide_v2,targetClass2id_mergeProp

targetid2class = {targetClass2id[one]:one for one in targetClass2id}

targetactid2class = {targetAct2id[one]:one for one in targetAct2id}

targetsingleactid2class = {targetSingleAct2id[one]:one for one in targetSingleAct2id}

eval_target = {
	"Vehicle":["car","motorcycle","bus","truck","vehicle"],
	"Person":"person",
}

eval_best = "Person" # not used anymore, we use average as the best metric



# load all ground truth into memory
def read_data_diva(config,idlst,framepath,annopath,tococo=False,randp=None,is_train=False):
	assert idlst is not None
	assert framepath is not None
	assert annopath is not None
	assert len(targetClass2id.keys()) == config.num_class

	# load the coco class name to classId so we could convert the label name to label classId
	if tococo:
		add_coco(config,config.datajson)

	imgs = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(idlst,"r").readlines()]

	if randp is not None:
		imgs = random.sample(imgs,int(len(imgs)*randp))

	data = {"imgs":[],"gt":[]}
	print "loading data.."
	if config.print_params:
		imgs = imgs[:100]
	# in diva dataset, some class may be ignored
	ignored_classes = {}
	
	targetClass2exist = {classname:0 for classname in targetClass2id}

	num_empty_actboxes = 0
	targetAct2exist = {classname:0 for classname in targetAct2id}
	ignored_act_classes = {}

	num_empty_single_actboxes = 0
	ignored_single_act_classes = {}
	targetAct2exist_single = {classname:0 for classname in targetSingleAct2id}
	act_single_fgratio = []

	if config.debug:
		imgs = imgs[:1000]

	if (config.train_skip > 1) and is_train:
		imgs.sort()
		ori_num = len(imgs)
		imgs = imgs[config.train_skip_offset::config.train_skip]
		print "skipping [%s::%s], got %s/%s"%(config.train_skip_offset,config.train_skip,len(imgs),ori_num)
	if (config.val_skip > 1) and not is_train:
		imgs.sort()
		ori_num = len(imgs)
		imgs = imgs[config.val_skip_offset::config.val_skip]
		print "skipping [%s::%s], got %s/%s"%(config.val_skip_offset,config.val_skip,len(imgs),ori_num)


	# get starts for each img, the label distribution
	label_dist = {classname:[] for classname in targetClass2id} # class -> [] num_box in each image
	label_dist_all = []

	for img in tqdm(imgs, ascii=True, smoothing=0.5):
		
		anno = os.path.join(annopath,"%s.npz"%img)
		if not os.path.exists(anno):
			continue
		anno = dict(np.load(anno)) # 'boxes' -> [K,4] 
		# boxes are x1,y1,x2,y2

		original_box_num = len(anno['boxes'])

		# feed act box as object boxes
		if config.act_as_obj:
			anno['labels'] = anno['actlabels']
			anno['boxes'] = anno['actboxes']

		# labels are one word, diva classname
		if tococo:
			# convert the classname to coco classid
			# some label is not used
			good_ids = []
			labels = []
			for i,classname in enumerate(list(anno['labels'])):
				targetClass2exist[classname] = 1
				if targetClass1to1[classname] is not None:
					coco_classname = targetClass1to1[classname]
					labels.append(config.class_to_classId[coco_classname])
					good_ids.append(i)
			anno['labels'] = labels
			anno['boxes'] = anno['boxes'][good_ids,:]
		else:
			labels = []
			boxes = []
			for i,classname in enumerate(list(anno['labels'])):
				if targetClass2id.has_key(classname):
					targetClass2exist[classname] = 1
					labels.append(targetClass2id[classname])
					boxes.append(anno['boxes'][i])
				else:
					ignored_classes[classname] = 1
			anno['boxes'] = np.array(boxes,dtype="float32")
			anno['labels'] = labels


		#assert len(anno['boxes']) > 0
		if len(anno['boxes']) == 0:
			continue
		assert len(anno['labels']) == len(anno['boxes']),(anno['labels'],anno['boxes'])
		assert anno['boxes'].dtype == np.float32

		# statics
		if config.show_stat:
			for classname in label_dist:
				num_box_this_img = len([l for l in labels if l == targetClass2id[classname]])
				label_dist[classname].append(num_box_this_img)
			label_dist_all.append(len(labels))

		if config.add_act:		
			# for activity anno, we couldn't remove any of the boxes
			assert len(anno['boxes']) == original_box_num
			if config.act_v2:
				# make multi class labels 
				# BG class is at index 0
				K = len(anno['boxes'])
				actSingleLabels = np.zeros((K,config.num_act_single_class),dtype="uint8")
				
				# use this to mark BG
				hasClass = np.zeros((K),dtype="bool")
				for i,classname in enumerate(list(anno['actSingleLabels'])):
					if targetSingleAct2id.has_key(classname):
						targetAct2exist_single[classname] = 1
						act_id = targetSingleAct2id[classname]
						box_id = anno['actSingleIdxs'][i]
						assert box_id >=0 and box_id < K
						actSingleLabels[box_id,act_id] = 1
						hasClass[box_id] = True
					else:
						ignored_single_act_classes[classname] = 1

				# mark the BG for boxes that has not activity annotation
				actSingleLabels[np.logical_not(hasClass), 0] = 1
				anno['actSingleLabels_npy'] = actSingleLabels
				
				# compute the BG vs FG ratio for the activity boxes
				act_single_fgratio.append(sum(hasClass)/float(K))

				if sum(hasClass) == 0:
					num_empty_single_actboxes+=1
					continue

			else:
				act_labels = []
				act_good_ids = []
				for i,classname in enumerate(list(anno['actlabels'])):
					if targetAct2id.has_key(classname):
						targetAct2exist[classname] = 1
						act_labels.append(targetAct2id[classname])
						act_good_ids.append(i)
					else:
						ignored_act_classes[classname] = 1
				#print anno['actboxes'].shape
				if anno['actboxes'].shape[0] == 0:# ignore this image
					num_empty_actboxes+=1
					continue
				anno['actboxes'] = anno['actboxes'][act_good_ids]
				anno['actboxidxs'] = anno['actboxidxs'][act_good_ids] # it is a npy array of python list, so no :
				anno['actlabels'] = act_labels
				assert len(anno['actboxes']) == len(anno['actlabels'])


		videoname = img.strip().split("_F_")[0]
		data['imgs'].append(os.path.join(framepath,videoname,"%s.jpg"%img))
		data['gt'].append(anno)

	print "loaded %s/%s data"%(len(data['imgs']),len(imgs))

	if config.show_stat:
		for classname in label_dist:
			d = label_dist[classname]
			ratios = [a/float(b) for a,b in zip(d, label_dist_all)]
			print "%s, [%s - %s], median %s per img, ratio:[%.3f - %.3f], median %.3f, no label %s/%s [%.3f]"%(classname, min(d), max(d), np.median(d), min(ratios), max(ratios), np.median(ratios), len([i for i in d if i==0]), len(d),len([i for i in d if i==0])/float(len(d)))
		print "each img has boxes: [%s - %s], median %s"%(min(label_dist_all),max(label_dist_all),np.median(label_dist_all),)


	if len(ignored_classes) > 0:
		print "ignored %s "%(ignored_classes.keys())
	noDataClasses = [classname for classname in targetClass2exist if targetClass2exist[classname] ==0]
	if len(noDataClasses) > 0:
		print "warning: class data not exists: %s, AR will be 1.0 for these"%(noDataClasses)
	if config.add_act:
		if config.act_v2:
			print " each frame positive act box percentage min %.4f, max %.4f, mean %.4f"%(min(act_single_fgratio),max(act_single_fgratio),np.mean(act_single_fgratio))
			if len(ignored_single_act_classes) > 0:
				print "ignored activity %s"%(ignored_single_act_classes.keys())
			print "%s/%s has no single activity boxes"%(num_empty_single_actboxes, len(data['imgs']))
			noDataClasses = [classname for classname in targetAct2exist_single if targetAct2exist_single[classname] ==0]
			if len(noDataClasses) > 0:
				print "warning: single activity class data not exists: %s, "%(noDataClasses)
		else:
			if len(ignored_act_classes) > 0:
				print "ignored activity %s"%(ignored_act_classes.keys())
			print "%s/%s has no activity boxes"%(num_empty_actboxes, len(data['imgs']))
			noDataClasses = [classname for classname in targetAct2exist if targetAct2exist[classname] ==0]
			if len(noDataClasses) > 0:
				print "warning: activity class data not exists: %s, "%(noDataClasses)


	return Dataset(data,add_gt=True)


# given the gen_gt_diva
# train on diva dataset
def train_diva(config):
	global eval_target,targetid2class,targetClass2id
	eval_target_weight = None
	if config.diva_class:
		# only care certain classes
		#eval_target = ["Vehicle","Person","Construction_Barrier","Door","Dumpster","Prop","Push_Pulled_Object","Bike","Parking_Meter"]
		eval_target = ["Vehicle","Person","Prop","Push_Pulled_Object","Bike"]
		eval_target = {one:1 for one in eval_target}
		eval_target_weight ={		
			"Person":0.15,
			"Vehicle":0.15,
			"Prop":0.15,
			"Push_Pulled_Object":0.15,
			"Bike":0.15,
		}

		if config.merge_prop:
			eval_target = ["Vehicle","Person","Prop","Push_Pulled_Object","Bike", "Prop_plus_Push_Pulled_Object"]
			eval_target = {one:1 for one in eval_target}
			eval_target_weight ={		
				"Person":0.15,
				"Vehicle":0.15,
				"Prop_plus_Push_Pulled_Object":0.2,
				"Bike":0.2,
				"Prop":0.15,
				"Push_Pulled_Object":0.15,
			}

	if config.add_act:
		# same for single box act
		act_eval_target = ["vehicle_turning_right","vehicle_turning_left","Unloading","Transport_HeavyCarry","Opening","Open_Trunk","Loading","Exiting","Entering","Closing_Trunk","Closing","Interacts","Pull","Riding","Talking","activity_carrying","specialized_talking_phone","specialized_texting_phone"] # "vehicle_u_turn" is not used since not exists in val set
		act_eval_target = {one:1 for one in act_eval_target}
		act_eval_target_weight ={one:1.0/len(act_eval_target) for one in act_eval_target}


	if config.act_as_obj:
		eval_target = ["vehicle_turning_right","vehicle_turning_left","Unloading","Transport_HeavyCarry","Opening","Open_Trunk","Loading","Exiting","Entering","Closing_Trunk","Closing","Interacts","Pull","Riding","Talking","activity_carrying","specialized_talking_phone","specialized_texting_phone"] # "vehicle_u_turn" is not used since not exists in val set

		

		eval_target = {one:1 for one in eval_target}
		eval_target_weight ={one:1.0/len(eval_target) for one in eval_target}
	
	self_summary_strs = Summary()
	stats = [] # tuples with {"metrics":,"step":,}
	# load the frame count data first
	
	train_data = read_data_diva(config,config.trainlst,config.imgpath,config.annopath,tococo=config.tococo,is_train=True) # True to filter data
	val_data = read_data_diva(config,config.vallst,config.valframepath,config.valannopath,tococo=False)#,randp=0.02)
	config.train_num_examples = train_data.num_examples

	if config.show_stat:
		sys.exit()

	# the total step (iteration) the model will run
	num_steps = int(math.ceil(train_data.num_examples/float(config.im_batch_size)))*config.num_epochs
	num_val_steps = int(math.ceil(val_data.num_examples/float(config.im_batch_size)))*1
	
	#config_vars = vars(config)
	#self_summary_strs.add("\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config_vars]))

	# model_per_gpu > 1 not work yet, need to set distributed computing
	#cluster = tf.train.ClusterSpec({"local": ["localhost:8000","localhost:8001"]})
	#server = tf.train.Server(cluster, job_name="local", task_index=0)
	#server = tf.train.Server(cluster, job_name="local", task_index=1)

	# two model, this is the lazy way
	#model = get_model(config) # input is image paths
	models = []
	gpuids = range(config.gpuid_start, config.gpuid_start+config.gpu)
	gpuids = gpuids * config.model_per_gpu 
	# example, model_per_gpu=2, gpu=2, gpuid_start=0
	gpuids.sort()# [0,0,1,1]
	taskids = range(config.model_per_gpu) * config.gpu # [0,1,0,1]

	for i,j in zip(gpuids,taskids):
		models.append(get_model(config,gpuid=i,task=j,controller=config.controller))

	config.is_train=False
	models_eval = []
	for i,j in zip(gpuids,taskids):
		models_eval.append(get_model(config,gpuid=i,task=j,controller=config.controller))
	config.is_train=True
	
	trainer = Trainer(models,config)
	tester = Tester(models_eval,config,add_mask=config.add_mask) # need final box and stuff?

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
		self_summary_strs.add("total parameters: %s"%(cal_total_param()))

		initialize(load=config.load,load_best=config.load_best,config=config,sess=sess)

		if config.print_params:
			for var in tf.global_variables():
				shape = var.get_shape()
				print "%s %s\n"%(var.name,shape)
			sys.exit()

		isStart = True

		loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss = -1,-1,-1,-1,-1

		#best_ar = (-1.0,1)# (AR,best_step)
		best = (-1.0,1) # (average_ar + average_ap)/2
		for batch in tqdm(train_data.get_batches(config.im_batch_size,num_batches=num_steps),total=num_steps,ascii=True,smoothing=1):

			global_step = sess.run(models[0].global_step) + 1 # start from 0 or the previous step
			
			validation_performance = None
			if (global_step % config.save_period == 0) or (config.load and isStart and (config.ignore_vars is None)): # time to save model

				tqdm.write("step:%s/%s (epoch:%.3f), this(last) step loss:%.6f, [%.6f,%.6f,%.6f,%.6f,]"%(global_step,num_steps,(config.num_epochs*global_step/float(num_steps)),loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss))
				tqdm.write("\tsaving model %s..."%global_step)
				saver.save(sess,os.path.join(config.save_dir,"model"),global_step=global_step)
				tqdm.write("\tdone")
				if config.skip_first_eval and isStart:
					tqdm.write("skipped first eval...")
					validation_performance = config.best_first

				else:
					
					e = {one:{} for one in eval_target.keys()} # cat_id -> imgid -> {"dm","dscores"}
					if config.add_act:
						e_act = {one:{} for one in act_eval_target.keys()} 

					for val_batch_ in tqdm(val_data.get_batches(config.im_batch_size,num_batches=num_val_steps,shuffle=False),total=num_val_steps,ascii=True,smoothing=1):
						batch_idx,val_batches = val_batch_
						this_batch_num = len(val_batches)
						# multiple image at a time for parallel inferencing with multiple gpu
						scales = []
						imgids = []
						for val_batch in val_batches:
							# load the image here and resize
							image = cv2.imread(val_batch.data['imgs'][0],cv2.IMREAD_COLOR)
							imgid = os.path.splitext(os.path.basename(val_batch.data['imgs'][0]))[0]
							imgids.append(imgid)
							assert image is not None,image
							image = image.astype("float32")
							val_batch.data['imgdata'] = [image]

							resized_image = resizeImage(image,config.short_edge_size,config.max_size)

							# rememember the scale and original image
							ori_shape = image.shape[:2]
							#print image.shape, resized_image.shape
							# average H/h and W/w ?
							scale = (resized_image.shape[0]*1.0/image.shape[0] + resized_image.shape[1]*1.0/image.shape[1])/2.0

							val_batch.data['resized_image'] = [resized_image]
							scales.append(scale)

						outputs = tester.step(sess,val_batch_)

						# post process this batch, also remember the ground truth
						for i in xrange(this_batch_num): # num gpu
							imgid = imgids[i]
							scale = scales[i]
							if config.add_act:
								if config.act_v2:
									boxes,labels,probs,actsingleboxes,actsinglelabels = outputs[i]
									actsingleboxes = actsingleboxes / scale
								else:
									boxes,labels,probs,actboxes,actlabels,actprobs = outputs[i]
									actboxes = actboxes / scale
							else:
								if config.add_mask:
									boxes,labels,probs,masks = outputs[i]
								else:
									boxes,labels,probs = outputs[i]
							
							val_batch = val_batches[i]

							boxes = boxes / scale

							target_dt_boxes = gather_dt(boxes,probs,labels,eval_target,targetid2class,tococo=config.tococo,coco_class_names=config.class_names)

							#gt
							anno = val_batch.data['gt'][0] # one val_batch is single image
							gt_boxes = gather_gt(anno['boxes'],anno['labels'],eval_target,targetid2class)


							# gt_boxes and target_dt_boxes for this image

							# eval on one single image
							match_dt_gt(e,imgid,target_dt_boxes,gt_boxes,eval_target)


							# eval the act box as well, put stuff in e_act
							if config.add_act and config.act_v2:
								# for v2, we have the single and pair boxes
								# actsingleboxes [K,4]
								# actsinglelabels [K,num_act_class]
								# first we filter the BG boxes

								topk=config.act_single_topk # we select topk act class for each box
								
								single_act_boxes,single_act_labels,single_act_probs = gather_act_singles(actsingleboxes,actsinglelabels,topk)

								target_act_dt_boxes = gather_dt(single_act_boxes,single_act_probs,single_act_labels,act_eval_target,targetsingleactid2class)

								# to collect the ground truth, each label will be a stand alone boxes
								anno = val_batch.data['gt'][0] # one val_batch is single image
								gt_single_act_boxes = []
								gt_single_act_labels = []
								gt_obj_boxes = anno['boxes']
								for bid,label in zip(anno['actSingleIdxs'],anno['actSingleLabels']):
									if label in act_eval_target:
										gt_single_act_boxes.append(gt_obj_boxes[bid])
										gt_single_act_labels.append(targetSingleAct2id[label])

								gt_act_boxes = gather_gt(gt_single_act_boxes,gt_single_act_labels,act_eval_target,targetsingleactid2class)

								match_dt_gt(e_act,imgid,target_act_dt_boxes,gt_act_boxes,act_eval_target)

							if config.add_act and not config.act_v2:							
								target_act_dt_boxes = gather_dt(actboxes,actprobs,actlabels,act_eval_target,targetactid2class)

								#gt
								
								anno = val_batch.data['gt'][0] # one val_batch is single image
								gt_act_boxes = gather_gt(anno['actboxes'],anno['actlabels'],act_eval_target,targetactid2class)

								# gt_boxes and target_dt_boxes for this image
								match_dt_gt(e_act,imgid,target_act_dt_boxes,gt_act_boxes,act_eval_target)

					# we have the dm and g matching for each image in e & e_act
					# max detection per image per category
					aps,ars = aggregate_eval(e,maxDet=100)

					aps_str = "|".join(["%s:%.5f"%(class_,aps[class_]) for class_ in aps])
					ars_str = "|".join(["%s:%.5f"%(class_,ars[class_]) for class_ in ars])
					#tqdm.write("\tval in %s at step %s, AP:%s, AR:%s, previous best AR for %s at %s is %.5f"%(num_val_steps,global_step,aps_str,ars_str,eval_best,best[1],best[0]))
					#validation_performance = ars[eval_best]
					# now we use average AR and average AP or weighted
					average_ap,average_ar = weighted_average(aps,ars,eval_target_weight)


					ap_weight = 0.5
					ar_weight = 0.5
					validation_performance = average_ap*ap_weight + average_ar*ar_weight

					if config.add_act:
						obj_validation_performance = validation_performance
						aps,ars = aggregate_eval(e_act,maxDet=100)

						act_aps_str = "|".join(["%s:%.5f"%(class_,aps[class_]) for class_ in aps])
						act_ars_str = "|".join(["%s:%.5f"%(class_,ars[class_]) for class_ in ars])
						
						average_ap,average_ar = weighted_average(aps,ars,act_eval_target_weight)


						ap_weight = 0.9
						ar_weight = 0.1
						act_validation_performance = average_ap*ap_weight + average_ar*ar_weight

						act_perf_weight = 0.5
						obj_perf_weight = 0.5
						validation_performance = obj_perf_weight*obj_validation_performance + act_perf_weight*act_validation_performance

						tqdm.write("\tval in %s at step %s, Obj AP:%s, AR:%s, obj performance %s"%(num_val_steps,global_step,aps_str,ars_str,obj_validation_performance))
						tqdm.write("\tAct AP:%s, AR:%s, this step val:%.5f, previous best val at %s is %.5f"%(act_aps_str,act_ars_str,validation_performance,best[1],best[0]))
					else:
						tqdm.write("\tval in %s at step %s, AP:%s, AR:%s, this step val:%.5f, previous best val at %s is %.5f"%(num_val_steps,global_step,aps_str,ars_str,validation_performance,best[1],best[0]))

				if validation_performance > best[0]:
					tqdm.write("\tsaving best model %s..."%global_step)
					bestsaver.save(sess,os.path.join(config.save_dir_best,"model"),global_step=global_step)
					tqdm.write("\tdone")
					best = (validation_performance,global_step)

				isStart = False
				if config.exit_after_val:
					print "exit after eval."
					break

			# skip if the batch is not complete, usually the last few ones
			if len(batch[1]) != config.gpu:
				continue

			try:
				loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss,train_op,act_losses = trainer.step(sess,batch)
			except Exception as e:
				print e
				bs = batch[1]
				print "batch files:%s"%([b.data['imgs'] for b in bs])
				sys.exit()
			
			if math.isnan(loss):
				tqdm.write("warning, nan loss: loss:%s,rpn_label_loss:%s, rpn_box_loss:%s, fastrcnn_label_loss:%s, fastrcnn_box_loss:%s"%(loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss))
				if config.add_act:
					tqdm.write("\tact_losses:%s"%(act_losses))
				print "batch:%s"%(batch[1][0].data['imgs'])
				sys.exit()

			# save these for ploting later
			stats.append({
				"s":float(global_step),
				"l":float(loss),
				"val":validation_performance
			})

		# save the last model
		if global_step % config.save_period != 0: # time to save model
			print "saved last model without evaluation."
			saver.save(sess,os.path.join(config.save_dir,"model"),global_step=global_step)
			
		if config.write_self_sum:
			self_summary_strs.writeTo(config.self_summary_path)

			with open(config.stats_path,"w") as f:
				json.dump(stats,f)


def pack(config):
	model = get_model(config)
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary 
	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)

		saver = tf.train.Saver()
		global_step = model.global_step
		# put input and output to a universal name for reference when in deployment
			# find the nessary stuff in model.get_feed_dict
		tf.add_to_collection("input",model.x)
		tf.add_to_collection("is_train",model.is_train) # TODO, change this to a constant 
		tf.add_to_collection("output",model.yp)
		# also save all the model config and note into the model
		pack_model_note = tf.get_variable("model_note",shape=[],dtype=tf.string,initializer=tf.constant_initializer(config.pack_model_note),trainable=False)
		full_config = tf.get_variable("model_config",shape=[],dtype=tf.string,initializer=tf.constant_initializer(json.dumps(vars(config))),trainable=False)

		print "saving packed model"
		# the following wont save the var model_note, model_config that's not in the graph, 
		# TODO: fix this
		"""
		# put into one big file to save
		input_graph_def = tf.get_default_graph().as_graph_def()
		#print [n.name for n in input_graph_def.node]
		 # We use a built-in TF helper to export variables to constants
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, # The session is used to retrieve the weights
			input_graph_def, # The graph_def is used to retrieve the nodes 
			[tf.get_collection("output")[0].name.split(":")[0]] # The output node names are used to select the usefull nodes
		) 
		output_graph = os.path.join(config.pack_model_path,"final.pb")
		# Finally we serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))
		"""
		# save it into a path with multiple files
		saver.save(sess,
			os.path.join(config.pack_model_path,"final"),
			global_step=global_step)
		print "model saved in %s"%(config.pack_model_path)

# given the box, extract feature
def boxfeat(config):
	imagelist = config.imgpath

	images = [line.strip() for line in open(config.imgpath,"r").readlines()]

	print "total images to test:%s"%len(images)

	if not os.path.exists(config.boxfeatpath):
		os.makedirs(config.boxfeatpath)

	model = get_model_boxfeat(config) # input image -> final_box, final_label, final_masks

	add_coco(config,config.datajson)
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1

		#count=0
		for image in tqdm(images,ascii=True):
			imagename = os.path.splitext(os.path.basename(image))[0]
			with open(os.path.join(config.boxjsonpath,imagename+".json"),"r") as f:
				this_data = json.load(f)
			boxes = np.asarray(this_data['boxes'],dtype="float") # should be a  [K,4] , x,y,w,h
			# -> x1,y1,x2,y2
			boxes[:, 2] = boxes[:,2] + boxes[:,0]
			boxes[:, 3] = boxes[:,3] + boxes[:,1]

	
			feed_dict = model.get_feed_dict(image,boxes)

			if config.boxclass:
				feature,label_probs = sess.run([model.feature,model.label_probs],feed_dict=feed_dict)
			else:
				feature, = sess.run([model.feature],feed_dict=feed_dict)

			assert len(feature) == len(boxes)

			# for debug
			"""
			print feature.shape
			print label_probs.shape

			for i,label_prob in enumerate(label_probs):
				print label_prob.shape
				label = np.argmax(label_prob)

				cat_name = config.class_names[label]

				ori_cat_name = this_data['cat_names'][i]
				ori_cat_id = config.class_to_classId[ori_cat_name]

				print "argmax label index:%s, cat_name:%s,logit:%s"%(label,cat_name,label_prob[label])
				if label == 0: # 0 is BG, let's get second largest
					label2 = label_prob.argsort()[-2:][::-1][1]
					print "argmax 2nd label index:%s, cat_name:%s,logit:%s"%(label2,config.class_names[label2],label_prob[label2])
				print "original label cat_name:%s,cat_id:%s,cur_logits:%s"%(ori_cat_name,ori_cat_id,label_prob[ori_cat_id])
			sys.exit()
			"""

			np.save(os.path.join(config.boxfeatpath,imagename+".npy"),feature)

# given the box/box_label/box_prob, extract the mask
def givenbox(config):
	imagelist = config.imgpath

	images = [line.strip() for line in open(config.imgpath,"r").readlines()]

	print "total images to test:%s"%len(images)

	if not os.path.exists(config.outbasepath):
		os.makedirs(config.outbasepath)

	model = get_model_givenbox(config) 

	add_coco(config,config.datajson)
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True 

	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1
		for image in tqdm(images,ascii=True):
			imagename = os.path.splitext(os.path.basename(image))[0]
			with open(os.path.join(config.boxjsonpath,imagename+".json"),"r") as f:
				this_data = json.load(f) # this is the same output from mask rcnn

			def gather_data(box_list):
				boxes,box_labels,box_probs = [],[],[]
				for one in box_list:
					boxes.append(one['bbox'])
					box_probs.append(one['score'])
					box_classId = config.class_to_classId[one['cat_name']] # [0-80]
					assert box_classId > 0, one
					box_labels.append(box_classId)
				boxes = np.asarray(boxes,dtype="float")
				box_labels = np.asarray(box_labels,dtype="int")
				box_probs = np.asarray(box_probs,dtype="float")
				return boxes, box_labels,box_probs

			boxes, box_labels,box_probs = gather_data(this_data)
			
			if boxes.shape[0] == 0:
				continue
			# boxes should be a  [K,4] , x,y,w,h
			# -> x1,y1,x2,y2
			boxes[:, 2] = boxes[:,2] + boxes[:,0]
			boxes[:, 3] = boxes[:,3] + boxes[:,1]

	
			# resized the image and box input
			feed_dict,ori_shape,scale = model.get_feed_dict(image,boxes,box_labels,box_probs)

			
			final_boxes,final_labels,final_probs,final_masks = sess.run([model.final_boxes,model.final_labels,model.final_probs,model.final_masks],feed_dict=feed_dict)

			final_boxes = final_boxes / scale

			final_masks = [fill_full_mask(box,mask,ori_shape) for box,mask in zip(final_boxes,final_masks)]

			pred = []

			for box, prob, label, mask in zip(final_boxes,final_probs,final_labels,final_masks):
				box[2] -= box[0]
				box[3] -= box[1] # produce x,y,w,h output

				cat_id = config.classId_to_cocoId[label]

				# encode mask
				rle = None
				if config.add_mask:
					rle = cocomask.encode(np.array(mask[:,:,None],order="F"))[0]
					rle['counts'] = rle['counts'].decode("ascii")

				res = {
					"category_id":cat_id,
					"cat_name":config.class_names[label], #[0-80]
					"score":float(round(prob,4)),
					"bbox": list(map(lambda x:float(round(x,1)),box)),
					"segmentation":rle
				}
				pred.append(res)

			# save the data
			resultfile = os.path.join(config.outbasepath,"%s.json"%imagename)
			with open(resultfile,"w") as f:
				json.dump(pred,f)




# given a list of images, do the forward, save each image result separately
def forward(config):
	imagelist = config.imgpath

	if config.extract_feat:
		assert config.feat_path is not None
		assert config.is_fpn
		if not os.path.exists(config.feat_path):
			os.makedirs(config.feat_path)
		print "also extracting fpn features"

	all_images = [line.strip() for line in open(config.imgpath,"r").readlines()]

	if config.forward_skip > 1:
		all_images.sort()
		ori_num = len(all_images)
		all_images = all_images[::config.forward_skip]
		print "skiiping %s, got %s/%s"%(config.forward_skip, len(all_images), ori_num)

	print "total images to test:%s"%len(all_images)

	#model = get_model(config) # input image -> final_box, final_label, final_masks
	#tester = Tester(model,config,add_mask=config.add_mask)
	models = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models.append(get_model(config,i,controller=config.controller))

	model_final_boxes = [model.final_boxes for model in models]
	# [R]
	model_final_labels = [model.final_labels for model in models]
	model_final_probs = [model.final_probs for model in models]

	if config.extract_feat:
		model_feats = [model.fpn_feature for model in models]

	if config.add_mask:
		# [R,14,14]
		model_final_masks = [model.final_masks for model in models]

	if config.add_act:
		if config.act_v2:
			model_act_single_boxes = [model.act_single_boxes for model in models]
			model_act_single_label_logits = [model.act_single_label_logits for model in models]
		else:
			model_act_final_boxes = [model.act_final_boxes for model in models]
			# [R]
			model_act_final_labels = [model.act_final_labels for model in models]
			model_act_final_probs = [model.act_final_probs for model in models]

	if not config.diva_class:# and not config.diva_class2:
		add_coco(config,config.datajson)

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	if not config.use_all_mem:
		tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1

		#count=0
		for images in tqdm(grouper(all_images,config.im_batch_size),ascii=True):
			images = [im for im in images if im is not None]
			# multigpu will need full image inpu
			this_batch_len = len(images)
			if this_batch_len != config.im_batch_size:
				need = config.im_batch_size - this_batch_len
				images.extend(all_images[:need]) # redo some images
			scales = []
			resized_images = []
			ori_shapes = []
			imagenames = []
			feed_dict = {}
			for i,image in enumerate(images):
				im = cv2.imread(image,cv2.IMREAD_COLOR)
				imagename = os.path.splitext(os.path.basename(image))[0]
				imagenames.append(imagename)

				ori_shape = im.shape[:2]

				# need to resize here, otherwise
				# InvalidArgumentError (see above for traceback): Expected size[1] in [0, 83], but got 120 [[Node: anchors/fm_anchors = Slice[Index=DT_INT32, T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](anchors/all_anchors, anchors/fm_anchors/begin, anchors/stack)]]

				resized_image = resizeImage(im,config.short_edge_size,config.max_size)

				scale = (resized_image.shape[0]*1.0/im.shape[0] + resized_image.shape[1]*1.0/im.shape[1])/2.0

				resized_images.append(resized_image)
				scales.append(scale)
				ori_shapes.append(ori_shape)
	
				feed_dict.update(models[i].get_feed_dict_forward(resized_image))

			sess_input = []

			if config.just_feat:
				outputs = sess.run(model_feats,feed_dict=feed_dict)
				for i,feat in enumerate(outputs):
					imagename = imagenames[i]
					
					featfile = os.path.join(config.feat_path, "%s.npy"%imagename)
					np.save(featfile, feat)

				continue # no bounding boxes

			if config.add_mask:
				for _,boxes,labels,probs,masks in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs,model_final_masks):
					sess_input+=[boxes,labels,probs,masks]
			else:	
				if config.add_act:
					if config.act_v2:
						for _,boxes,labels,probs,actboxes,actlabels in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs,model_act_single_boxes,model_act_single_label_logits):
							sess_input+=[boxes,labels,probs,actboxes,actlabels]
					else:
						for _,boxes,labels,probs,actboxes,actlabels,actprobs in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs,model_act_final_boxes,model_act_final_labels,model_act_final_probs):
							sess_input+=[boxes,labels,probs,actboxes,actlabels,actprobs]
				else:
					if config.extract_feat:
						for _,boxes,labels,probs,feats in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs,model_feats):
							sess_input+=[boxes,labels,probs,feats]
					else:
						for _,boxes,labels,probs in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs):
							sess_input+=[boxes,labels,probs]

			outputs = sess.run(sess_input,feed_dict=feed_dict)
			if config.add_mask:
				pn = 4
			else:
				pn = 3
				if config.add_act:
					pn=6
					if config.act_v2:
						pn=5
				else:
					if config.extract_feat:
						pn=4
			outputs = [outputs[i*pn:(i*pn+pn)] for i in xrange(len(images))]	

			for i,output in enumerate(outputs):
				scale = scales[i]
				ori_shape = ori_shapes[i]
				imagename = imagenames[i]
				if config.add_mask:
					final_boxes, final_labels, final_probs, final_masks = output
					final_boxes = final_boxes / scale
					final_masks = [fill_full_mask(box,mask,ori_shape) for box,mask in zip(final_boxes,final_masks)]
				else:
					if config.add_act:
						if config.act_v2:
							final_boxes, final_labels, final_probs,actsingleboxes,actsinglelabels  = output
							actsingleboxes = actsingleboxes / scale
						else:
							final_boxes, final_labels, final_probs,actboxes,actlabels,actprobs  = output
							actboxes = actboxes / scale
					else:
						if config.extract_feat:
							final_boxes, final_labels, final_probs, final_feat = output
							#print final_feats.shape# [1,7,7,256]
							# save the features
							
							featfile = os.path.join(config.feat_path, "%s.npy"%imagename)
							np.save(featfile, final_feat)
						else:
							final_boxes, final_labels, final_probs = output
					final_boxes = final_boxes / scale
					final_masks = [None for one in final_boxes]	

				pred = []

				for j,(box, prob, label, mask) in enumerate(zip(final_boxes,final_probs,final_labels,final_masks)):
					box[2] -= box[0]
					box[3] -= box[1] # produce x,y,w,h output

					if config.diva_class:# or config.diva_class2:
						cat_id = label
						cat_name = targetid2class[cat_id]
					else:
						cat_id = config.classId_to_cocoId[label]
						cat_name = config.class_names[label]

					# encode mask
					rle = None
					if config.add_mask:
						rle = cocomask.encode(np.array(mask[:,:,None],order="F"))[0]
						rle['counts'] = rle['counts'].decode("ascii")

					res = {
						"category_id":cat_id,
						"cat_name":cat_name, #[0-80]
						"score":float(round(prob,4)),
						"bbox": list(map(lambda x:float(round(x,1)),box)),
						"segmentation":rle,
					}


					pred.append(res)

				if config.add_act:
					act_pred = []

					if config.act_v2:
						# assemble the single boxes and pair boxes? 
						topk=config.act_single_topk
						single_act_boxes,single_act_labels,single_act_probs = gather_act_singles(actsingleboxes,actsinglelabels,topk)

						for j,(act_box, act_prob, act_label) in enumerate(zip(single_act_boxes,single_act_probs,single_act_labels)):
							act_box[2] -= act_box[0]
							act_box[3] -= act_box[1]
							act_name = targetsingleactid2class[act_label]
							res = {
								"category_id":act_label,
								"cat_name":act_name, 
								"score":float(round(act_prob,4)),
								"bbox": list(map(lambda x:float(round(x,1)),act_box)),
								"segmentation":None,
								"v2":1,
								"single":1,
							}
							act_pred.append(res)

					else:
						for j,(act_box, act_prob, act_label) in enumerate(zip(actboxes,actprobs,actlabels)):
							act_box[2] -= act_box[0]
							act_box[3] -= act_box[1]
							act_name = targetactid2class[act_label]
							res = {
								"category_id":act_label,
								"cat_name":act_name, 
								"score":float(round(act_prob,4)),
								"bbox": list(map(lambda x:float(round(x,1)),act_box)),
								"segmentation":None,
								"v2":0,
							}
							act_pred.append(res)


					# save the act data
					resultfile = os.path.join(config.actoutbasepath,"%s.json"%imagename)
					with open(resultfile,"w") as f:
						json.dump(act_pred,f)


				# save the data
				resultfile = os.path.join(config.outbasepath,"%s.json"%imagename)
				with open(resultfile,"w") as f:
					json.dump(pred,f)

from glob import glob
# only get fpn backbone feature for each video, no object detection
def videofeat(config):
	assert config.feat_path is not None
	assert config.is_fpn
	assert config.videolst is not None
	if not os.path.exists(config.feat_path):
		os.makedirs(config.feat_path)

	# imgpath is the frame path,
	# need videolst
	# we get all the image first
	print "getting imglst..."
	imgs = {}# videoname -> frames
	total=0
	for videoname in [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(config.videolst).readlines()]:
		framepath = os.path.join(config.imgpath, "%s"%videoname)
		frames = glob(os.path.join(framepath, "*.jpg"))
		frames.sort()
		frames = frames[::config.forward_skip] # some only have 1-3 frames
		imgs[videoname] = frames
		total+=len(frames)
	print "done, got %s imgs"%total


	#model = get_model(config) # input image -> final_box, final_label, final_masks
	#tester = Tester(model,config,add_mask=config.add_mask)
	models = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models.append(get_model(config,i,controller=config.controller))

	
	model_feats = [model.fpn_feature for model in models]

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	if not config.use_all_mem:
		tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1

		#count=0
		for videoname in tqdm(imgs,ascii=True):
			if config.skip:
				if os.path.exists(os.path.join(config.feat_path,"%s.npy"%videoname)):
					continue
			feats = []
			for images in tqdm(grouper(imgs[videoname],config.im_batch_size),ascii=True):
				images = [im for im in images if im is not None]
				# multigpu will need full image inpu
				this_batch_len = len(images)
				need=0
				if this_batch_len != config.im_batch_size:
					need = config.im_batch_size - this_batch_len
					repeats = [imgs[videoname][0] for i in xrange(need)]
					images.extend(repeats) # redo some images

				feed_dict = {}
				for i,image in enumerate(images):
					im = cv2.imread(image,cv2.IMREAD_COLOR)

					resized_image = resizeImage(im,config.short_edge_size,config.max_size)
					
					feed_dict.update(models[i].get_feed_dict_forward(resized_image))
				sess_input = []

				outputs = sess.run(model_feats,feed_dict=feed_dict)
				this_feats = []
				for i,feat in enumerate(outputs[:len(outputs)-need]): # ignore the repeated ones
					
					this_feats.append(feat)
				assert len(this_feats) == this_batch_len
				feats.extend(this_feats)

			feats = np.array(feats)
			#(380, 1, 7, 7, 256) 
			feats = np.squeeze(feats,axis=1)
			feat_file = os.path.join(config.feat_path,"%s.npy"%videoname)
			np.save(feat_file, feats)



def read_data_coco(datajson,config,add_gt=False,load_coco_class=False):

	with open(datajson,"r") as f:
		dj = json.load(f)

	if load_coco_class:
		add_coco(config,datajson)


	data = {"imgs":[],'ids':[]}
	if add_gt:
		data = {"imgs":[],'ids':[],"gt":[]}

	# read coco annotation file
	for one in dj['images']:
		imgid = int(one['id'])
		imgfile = os.path.join(config.imgpath,one['file_name'])
		if config.coco2014_to_2017:
			imgfile = os.path.join(config.imgpath,one['file_name'].split("_")[-1])
		data['imgs'].append(imgfile)
		data['ids'].append(imgid)
		if add_gt:
			# load the bounding box and so on
			pass


	return Dataset(data,add_gt=add_gt)


# for testing, dataset -> {"imgs":[],'ids':[]}, imgs is the image file path,
def forward_coco(dataset,num_batches,config,sess,tester,resize=True):
	assert not config.diva_class # not working for this yet
	# "id" -> (boxes, probs, labels, masks)
	#pred = {}
	# each is (image_id,cat_id,bbox,score,segmentation)
	pred = []
	for evalbatch in tqdm(dataset.get_batches(config.im_batch_size,num_batches=num_batches,shuffle=False,cap=True),total=num_batches):

		_, batches = evalbatch

		scales = []
		ori_shapes = []
		image_ids = []
		for batch in batches:
			# load the image here and resize
			image = cv2.imread(batch.data['imgs'][0],cv2.IMREAD_COLOR)
			assert image is not None,batch.data['imgs'][0]
			image = image.astype("float32")
			imageId = batch.data['ids'][0]
			image_ids.append(imageId)
			batch.data['imgdata'] = [image]
			#if imageId != 139:
			#	continue

			# resize image
			# ppwwyyxx's code do resizing in eval
			if resize:
				resized_image = resizeImage(image,config.short_edge_size,config.max_size)
			else:
				resized_image = image

			# rememember the scale and original image
			ori_shape = image.shape[:2]
			#print image.shape, resized_image.shape
			# average H/h and W/w ?
			scale = (resized_image.shape[0]*1.0/image.shape[0] + resized_image.shape[1]*1.0/image.shape[1])/2.0

			batch.data['resized_image'] = [resized_image]
			scales.append(scale)
			ori_shapes.append(ori_shape)

		outputs = tester.step(sess,evalbatch) 
		
		for i,output in enumerate(outputs):
			scale = scales[i]
			ori_shape = ori_shapes[i]
			imgid = image_ids[i]
			if config.add_mask:
				final_boxes, final_labels, final_probs, final_masks = output
				final_boxes = final_boxes / scale
				final_masks = [fill_full_mask(box,mask,ori_shape) for box,mask in zip(final_boxes,final_masks)]
			else:
				final_boxes, final_labels, final_probs = output
				final_boxes = final_boxes / scale
				final_masks = [None for one in final_boxes]

			for box, prob, label, mask in zip(final_boxes,final_probs,final_labels,final_masks):
				box[2] -= box[0]
				box[3] -= box[1]

				cat_id = config.classId_to_cocoId[label]

				# encode mask
				rle = None
				if config.add_mask:
					rle = cocomask.encode(np.array(mask[:,:,None],order="F"))[0]
					rle['counts'] = rle['counts'].decode("ascii")

				res = {
					"image_id":imgid,#int
					"category_id":cat_id,
					"cat_name":config.class_names[label], #[0-80]
					"score":float(round(prob,4)),
					"bbox": list(map(lambda x:float(round(x,1)),box)),
					"segmentation":rle
				}
				pred.append(res)

		#print [(one['category_id'],one['score'],one['bbox']) for one in pred]
		#print imageId
		#sys.exit()

	return pred


# test on coco dataset
def test(config):
	test_data = read_data_coco(config.datajson,config=config,add_gt=False,load_coco_class=True)
	
	print "total testing samples:%s"%test_data.num_examples
	
	#model = get_model(config) # input image -> final_box, final_label, final_masks
	#tester = Tester(model,config,add_mask=config.add_mask)
	models = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models.append(get_model(config,i,controller=config.controller))
	tester = Tester(models,config,add_mask=config.add_mask)


	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	if not config.use_all_mem:
		tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=tfconfig) as sess:

		initialize(load=True,load_best=config.load_best,config=config,sess=sess)
		# num_epoch should be 1
		assert config.num_epochs == 1
		num_steps = int(math.ceil(test_data.num_examples/float(config.im_batch_size)))*config.num_epochs

		# a list of imageids
		pred = forward_coco(test_data,num_steps,config,sess,tester,resize=True)

		#with open("coco.json","w") as f:
		#	json.dump(pred,f)
		if config.use_coco_eval:

			evalcoco(pred,config.datajson,add_mask=config.add_mask)

		else:

			# check our AP implementation, use our map implementation
			# load the annotation first
			all_cat_ids = {}
			with open(config.datajson,"r") as f:
				data = json.load(f)
			gt = {} # imageid -> boxes:[], catids
			for one in data['annotations']:
				cat_id = one['category_id']
				all_cat_ids[cat_id] = 1
				imageid = int(one['image_id'])
				if not gt.has_key(imageid):
					gt[imageid] = {} # cat_ids -> boxes[]
				#gt[imageid]['boxes'].append(one['bbox']) # (x,y,w,h), float
				#gt[imageid]['cat_ids'].append(one['category_id'])
				if not gt[imageid].has_key(cat_id):
					gt[imageid][cat_id] = []
				gt[imageid][cat_id].append(one['bbox'])


			print "total category:%s"%len(all_cat_ids)

			# get the aps/ars for each frame
			dt = {} # imageid -> cat_id -> {boxes,scores}
			for one in pred:
				imageid = one['image_id']
				dt_bbox = one['bbox']
				score = one['score']
				cat_id = one['category_id']
				if not dt.has_key(imageid):
					dt[imageid] = {}
				if not dt[imageid].has_key(cat_id):
					dt[imageid][cat_id] = []
				dt[imageid][cat_id].append((dt_bbox,score))

			# get eval for each image
			"""
			aps = {class_:[] for class_ in all_cat_ids}
			ars = {class_:[] for class_ in all_cat_ids}
			for imageid in gt:
				for cat_id in gt[imageid]:
					if not dt.has_key(imageid):
						ars[cat_id].append(0.0)
					else:
						d = []
						dscores = []
						if dt[imageid].has_key(cat_id):
							# sort the boxes based on the score first
							dt[imageid][cat_id].sort(key=operator.itemgetter(1),reverse=True)
							for boxes,score in dt[imageid][cat_id]:
								d.append(boxes)
								dscores.append(score)
							
						g = gt[imageid][cat_id]

						dm,gm = match_detection(d,g,cocomask.iou(d,g,[0 for _ in xrange(len(g))]),iou_thres=0.5)

						ap = computeAP(dm)
						ar = computeAR(dm,gm,recall_k=10)

						aps[cat_id].append(ap)
						ars[cat_id].append(ar)
			# aggregate the aps and ars
			aps = [sum(aps[cat_id])/float(len(aps[cat_id])) for cat_id in aps.keys()]
			ars = [sum(ars[cat_id])/float(len(ars[cat_id])) for cat_id in ars.keys()]
			mean_ap = sum(aps)/len(aps)
			mean_ar = sum(ars)/len(ars)
			"""
			# accumulate all detection and compute AP once
			e = {} # imageid -> catid
			start = time.time()
			for imageid in gt:
				e[imageid] = {}
				for cat_id in gt[imageid]:
					g = gt[imageid][cat_id]

					e[imageid][cat_id] = {
						"dscores":[],
						"dm":[],
						"gt_num":len(g),
					}
				
					d = []
					dscores = []
					if dt.has_key(imageid) and dt[imageid].has_key(cat_id):
						# sort the boxes based on the score first
						dt[imageid][cat_id].sort(key=operator.itemgetter(1),reverse=True)
						for boxes,score in dt[imageid][cat_id]:
							d.append(boxes)
							dscores.append(score)
						
					
					dm,gm = match_detection(d,g,cocomask.iou(d,g,[0 for _ in xrange(len(g))]),iou_thres=0.5)

					e[imageid][cat_id]['dscores'] = dscores
					e[imageid][cat_id]['dm'] = dm

			# accumulate results
			maxDet = 100 # max detection per image per category
			aps = {}
			ars = {}
			for catId in all_cat_ids:
				# put all detection scores from all image together
				dscores = np.concatenate([e[imageid][catId]['dscores'][:maxDet] for imageid in e if e[imageid].has_key(catId)])
				# sort
				inds = np.argsort(-dscores,kind="mergesort")
				dscores_sorted = dscores[inds]

				# put all detection annotation together based on the score sorting
				dm = np.concatenate([e[imageid][catId]['dm'][:maxDet] for imageid in e if e[imageid].has_key(catId)])[inds]
				num_gt = np.sum([e[imageid][catId]['gt_num'] for imageid in e if e[imageid].has_key(catId)])

				aps[catId] = computeAP(dm)
				ars[catId] = computeAR_2(dm,num_gt)

			mean_ap = np.mean([aps[catId] for catId in aps])
			mean_ar = np.mean([ars[catId] for catId in ars])
			took = time.time() - start
			print "total dt image:%s, gt image:%s"%(len(dt),len(gt))

			print "mean AP with IoU 0.5:%s, mean AR with max detection %s:%s, took %s seconds"%(mean_ap,maxDet,mean_ar,took)




def initialize(load,load_best,config,sess):
	tf.global_variables_initializer().run()
	if load:
		print "restoring model..."
		allvars = tf.global_variables()
		allvars = [var for var in allvars if "global_step" not in var.name]
		#restore_vars = allvars
		opts = ["Adam","beta1_power","beta2_power","Adam_1","Adadelta_1","Adadelta","Momentum"]
		allvars = [var for var in allvars if var.name.split(":")[0].split("/")[-1] not in opts]
		# so allvars is actually the variables except things for training

		if config.ignore_vars is not None:
			ignore_vars = config.ignore_vars.split(":")
			ignore_vars.extend(opts)
			# also these
			#ignore_vars+=["global_step"]

			restore_vars = []
			for var in allvars:
				ignore_it = False
				for ivar in ignore_vars:
					if ivar in var.name:
						ignore_it=True
						print "ignored %s"%var.name
						break
				if not ignore_it:
					restore_vars.append(var)

			print "ignoring %s variables, original %s vars, restoring for %s vars"% (len(ignore_vars),len(allvars),len(restore_vars))

		else:
			restore_vars = allvars

		saver = tf.train.Saver(restore_vars, max_to_keep=5)

		load_from = None
		
		if config.load_from is not None:
			load_from = config.load_from
		else:
			if load_best:
				load_from = config.save_dir_best
			else:
				load_from = config.save_dir
		
		ckpt = tf.train.get_checkpoint_state(load_from)
		if ckpt and ckpt.model_checkpoint_path:
			loadpath = ckpt.model_checkpoint_path
					
			saver.restore(sess, loadpath)
			print "Model:"
			print "\tloaded %s"%loadpath
			print ""
		else:
			if os.path.exists(load_from): 
				if load_from.endswith(".ckpt"):
					# load_from should be a single .ckpt file
					saver.restore(sess,load_from)
				elif load_from.endswith(".npz"):
					# load from dict
					weights = np.load(load_from)
					params = {get_op_tensor_name(n)[1]:v for n,v in dict(weights).iteritems()}
					param_names = set(params.iterkeys())

					#variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

					variables = restore_vars

					variable_names = set([k.name for k in variables])

					intersect = variable_names & param_names

					restore_vars = [v for v in variables if v.name in intersect]

					with sess.as_default():
						for v in restore_vars:
							vname = v.name
							v.load(params[vname])

					#print variables # all the model's params

					not_used = [(one,weights[one].shape) for one in weights.keys() if get_op_tensor_name(one)[1] not in intersect]
					if len(not_used) > 0:
						print "warning, %s/%s in npz not restored:%s"%(len(weights.keys()) - len(intersect), len(weights.keys()), not_used)

					if config.show_restore:			
						print "loaded %s vars:%s"%(len(intersect),intersect)
						

				else:
					raise Exception("Not recognized model type:%s"%load_from)
			else:
				raise Exception("Model not exists")
		print "done."


# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def cal_total_param():
	total = 0
	for var in tf.trainable_variables():
		shape = var.get_shape()
		var_num = 1
		for dim in shape:
			var_num*=dim.value
		total+=var_num
	return total


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

if __name__ == "__main__":
	config = get_args()

	if config.is_pack_model:
		pack(config)
	else:
		if config.mode == "train":
			train_diva(config)
		elif config.mode == "test":
			test(config)
		elif config.mode == "forward":
			forward(config)
		elif config.mode == "boxfeat": # given image list and each image's box, extract CNN feature
			boxfeat(config)
		elif config.mode == "givenbox":
			givenbox(config) # given image, boxes, get the mask output
		elif config.mode == "videofeat":
			videofeat(config)
		else:
			raise Exception("mode %s not supported"%(config.mode))
