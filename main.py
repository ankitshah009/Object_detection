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
from utils import evalcoco,get_op_tensor_name,match_detection,computeAP,computeAR,computeAR_2,grouper

from utils import Dataset,Summary

def get_args():
	global targetClass2id, targetid2class
	parser = argparse.ArgumentParser()

	parser.add_argument("datajson")
	parser.add_argument("imgpath")

	parser.add_argument("--outbasepath",type=str,default=None,help="full path will be outbasepath/modelname/runId")

	parser.add_argument("--modelname",type=str,default=None)
	parser.add_argument("--num_class",type=int,default=81,help="num catagory + 1 background")

	parser.add_argument("--tococo",action="store_true",help="for training in diva using coco model, map diva class1to1 to coco")
	parser.add_argument("--diva_class",action="store_true",help="the last layer is 16 (full) class output as the diva object classes")
	#parser.add_argument("--diva_class2",action="store_true",help="the last layer is 16 class output as the diva object classes")

	parser.add_argument("--add_act",action="store_true",help="add activitiy model")
	parser.add_argument("--num_act_class",type=int,default=36,help="num catagory + 1 background")

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
	parser.add_argument("--finer_resolution",action="store_true",help="fpn use finer resolution conv")
	parser.add_argument("--fix_fpn_model",action="store_true",help="for finetuneing a fpn model, whether to fix the lateral and poshoc weights")
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
	
	parser.add_argument("--rpn_test_post_nms_topk",type=int,default=700,help="test post nms, input to fast rcnn")

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

	#assert args.gpu > 1
	assert args.gpu == args.im_batch_size # one gpu one image
	args.controller = "/cpu:0"

	assert int(args.diva_class) + int(args.tococo) == 1

	targetid2class = targetid2class
	targetClass2id = targetClass2id
	#if args.diva_class2:
	#	targetid2class = targetid2class_v2
	#	targetClass2id = targetClass2id_v2

	if not args.tococo and ((args.mode == "train") or (args.mode == "test")):
		assert args.num_class == len(targetClass2id.keys())

	if args.vis_pre:
		assert args.vis_path is not None
		if not os.path.exists(args.vis_path):
			os.makedirs(args.vis_path)

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

	if args.is_cascade_rcnn:
		assert args.is_fpn
		args.cascade_num_stage = 3
		args.cascade_ious = [0.5, 0.6, 0.7]
		
		args.cascade_bbox_reg = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]


	if args.is_fpn:
		args.anchor_strides = (4, 8, 16, 32, 64)

		args.fpn_resolution_requirement = float(args.anchor_strides[3]) # [3] is 32, since we build FPN with r2,3,4,5?

		if args.finer_resolution:
			args.anchor_strides = (2, 4, 8, 16, 32)
			args.fpn_resolution_requirement = float(args.anchor_strides[4])

		
		args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * args.fpn_resolution_requirement

		args.fpn_num_channel = 256

		args.fpn_frcnn_fc_head_dim = 1024

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
	args.anchor_sizes = (32,64,128,256,512)
	if args.small_anchor_exp:
		args.anchor_sizes = (16,32,64,96,128,256) # not used for fpn

	if args.finer_resolution:
		args.anchor_sizes = (16, 32,64,128,256)

	args.anchor_ratios = (0.5,1,2)
	args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
	# iou thres to determine anchor label
	#args.positive_anchor_thres = 0.7
	#args.negative_anchor_thres = 0.3

	# when getting region proposal, avoid getting too large boxes
	args.bbox_decode_clip = np.log(args.max_size / 16.0)

	# RPN training
	args.rpn_fg_ratio = 0.5
	args.rpn_batch_per_im = args.rpn_batch_size
	args.rpn_min_size = 0
	args.rpn_proposal_nms_thres = 0.7
	args.rpn_train_pre_nms_topk = 12000
	args.rpn_train_post_nms_topk = 2000

	# fastrcnn
	args.fastrcnn_batch_per_im = args.frcnn_batch_size
	args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype='float32')
	#if args.is_fpn:
	#	args.fastrcnn_bbox_reg_weights = np.array([20, 20, 10, 10], dtype='float32')
	args.fastrcnn_fg_thres = 0.5 # iou thres
	#args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

	# testing
	args.rpn_test_pre_nms_topk = 6000
	#args.rpn_test_post_nms_topk = 700 #1300 # 700 takes 40 hours, # OOM at 1722,28,28,1024 # 800 OOM for gpu4
	args.fastrcnn_nms_thres = 0.5
	args.fastrcnn_nms_iou_thres = 0.5

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


# diva -> coco
# all diva class has to be covered
"""
targetClass = {
	"Vehicle":["car","motorcycle","bus","truck"],
	"Person":["person"],
	"Parking_Meter":["parking meter"],
	"Tree":[],
	"Other":[],
	"Trees":[],
	"Construction_Barrier":[],
	"Door":[],
	"Dumpster":[],
	"Push_Pulled_Object":[],
	"Construction_Vehicle":["truck"],
	"Prop":[],
	"Bike":['bicycle'],
	"Animal":['dog','cat']
}
"""
# for using a COCO model to finetuning with DIVA data.
targetClass1to1 = {
	"Vehicle":"car",
	"Person":"person",
	"Parking_Meter":"parking meter",
	"Tree":"potted plant",
	"Other":None,
	"Trees":"potted plant",
	"Construction_Barrier":None,
	"Door":None,
	"Dumpster":None,
	"Push_Pulled_Object":"suitcase", # should be a dolly
	"Construction_Vehicle":"truck",
	"Prop":"handbag",
	"Bike":'bicycle',
	"Animal":'dog',
	"Articulated_Infrastructure":None,
}
""" # the number of bbox in DIVA training set
Vehicle:3618465
Person:809684
Tree:686551
Other:303851
Trees:227180
Construction_Barrier:146712
Parking_Meter:103627
Door:81362
Dumpster:68441
Push_Pulled_Object:44788
Construction_Vehicle:16981
Prop:14441
Bike:13031
Animal:4055
"""

targetClass2id = { 
	"BG":0,
	"Vehicle":1,
	"Person":2,
	"Parking_Meter":3,
	"Tree":4,
	"Other":5,
	"Trees":6,
	"Construction_Barrier":7,
	"Door":8,
	"Dumpster":9,
	"Push_Pulled_Object":10,
	"Construction_Vehicle":11,
	"Prop":12,
	"Bike":13,
	"Animal":14,
	"Articulated_Infrastructure":15,
}
targetAct2id = {
	"BG":0,
	"activity_walking": 1,
	"vehicle_moving": 2,
	"activity_standing": 3,
	"vehicle_stopping": 4,
	"activity_carrying": 5,
	"vehicle_starting": 6,
	"vehicle_turning_right": 7,
	"vehicle_turning_left": 8,
	"activity_gesturing": 9,
	"Closing": 10,
	"Opening": 11,
	"Interacts": 12,
	"Exiting": 13,
	"Entering": 14, # 3, 0.014
	"Talking": 15, # (4, '0.045'), (3, '0.224')
	"Transport_HeavyCarry": 16, # (3, '0.156')
	"Unloading": 17, # (4, '0.250'), (2, '0.273'), (3, '0.477')
	"Pull": 18,
	"Loading": 19, # (4, '0.132'), (2, '0.342'), (3, '0.526')
	"Open_Trunk": 20, # (3, '0.114')
	"Closing_Trunk": 21, # (3, '0.194')
	"Riding": 22,
	"specialized_texting_phone": 23,
	"Person_Person_Interaction": 24,
	"specialized_talking_phone": 25,
	"activity_running": 26,
	#"specialized_miscellaneous": 0,
	"vehicle_u_turn": 27,
	"PickUp": 28, # [(3, '0.364'), (2, '0.636')]
	"specialized_using_tool": 29,
	"SetDown": 2, # [(4, '0.100'), (3, '0.400'), (2, '0.500')]
	"activity_crouching": 30,
	"activity_sitting": 31,
	"Object_Transfer": 32, #[(2, '0.375'), (3, '0.625')]
	"Push": 33,
	"PickUp_Person_Vehicle": 34,
	#"Misc": 0,
	"DropOff_Person_Vehicle": 35,
	#"Drop": 0,
	#"specialized_umbrella":0,
}
"""
targetClass2id_v2 = { 
	"BG":0,
	"Vehicle":1,
	"Person":2,
	"Parking_Meter":3,
	"Construction_Barrier":4,
	"Door":5,
	"Push_Pulled_Object":6,
	"Construction_Vehicle":7,
	"Prop":8,
	"Bike":9,
}
"""

targetid2class = {targetClass2id[one]:one for one in targetClass2id}

targetactid2class = {targetAct2id[one]:one for one in targetAct2id}


eval_target = {
	"Vehicle":["car","motorcycle","bus","truck","vehicle"],
	"Person":"person",
}

eval_best = "Person" # not used anymore, we use average as the best metric

def aggregate_eval(e,maxDet=100):
	aps = {}
	ars = {}
	for catId in e:
		e_c = e[catId]
		# put all detection scores from all image together
		dscores = np.concatenate([e_c[imageid]['dscores'][:maxDet] for imageid in e_c])
		# sort
		inds = np.argsort(-dscores,kind="mergesort")
		dscores_sorted = dscores[inds]

		# put all detection annotation together based on the score sorting
		dm = np.concatenate([e_c[imageid]['dm'][:maxDet] for imageid in e_c])[inds]
		num_gt = np.sum([e_c[imageid]['gt_num'] for imageid in e_c])

		aps[catId] = computeAP(dm)
		ars[catId] = computeAR_2(dm,num_gt)
	return aps,ars
def weighted_average(aps,ars,eval_target_weight):

	if eval_target_weight is not None:
		average_ap = sum([aps[class_]*eval_target_weight[class_] for class_ in aps])
		average_ar = sum([ars[class_]*eval_target_weight[class_] for class_ in ars])
	else:
		average_ap = sum(aps.values())/float(len(aps))
		average_ar = sum(ars.values())/float(len(ars))

	return average_ap,average_ar


# load all ground truth into memory
def read_data_diva(config,idlst,framepath,annopath,tococo=False,randp=None):
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
	num_empty_actboxes = 0
	targetClass2exist = {classname:0 for classname in targetClass2id}
	targetAct2exist = {classname:0 for classname in targetAct2id}
	for img in tqdm(imgs, ascii=True, smoothing=0.5):
		
		anno = os.path.join(annopath,"%s.npz"%img)
		if not os.path.exists(anno):
			continue
		anno = dict(np.load(anno)) # 'boxes' -> [K,4] 
		# boxes are x1,y1,x2,y2

		original_box_num = len(anno['boxes'])

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

		if config.add_act:
			ignored_act_classes = {}
			# for activity anno, we couldn't remove any of the boxes
			assert len(anno['boxes']) == original_box_num
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

			# TODO: filter for single, pairboxes


		videoname = img.strip().split("_F_")[0]
		data['imgs'].append(os.path.join(framepath,videoname,"%s.jpg"%img))
		data['gt'].append(anno)

	print "loaded %s/%s data"%(len(data['imgs']),len(imgs))
	if len(ignored_classes) > 0:
		print "ignored %s "%(ignored_classes.keys())
	noDataClasses = [classname for classname in targetClass2exist if targetClass2exist[classname] ==0]
	if len(noDataClasses) > 0:
		print "warning: class data not exists: %s, "%(noDataClasses)
	if config.add_act:
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
	global eval_target
	eval_target_weight = None
	if config.diva_class:
		# only care certain classes
		eval_target = ["Vehicle","Person","Construction_Barrier","Door","Dumpster","Prop","Push_Pulled_Object","Bike","Parking_Meter"]
		eval_target = {one:1 for one in eval_target}
		eval_target_weight ={		
			"Person":0.2,
			"Vehicle":0.15,
			"Prop":0.15,
			"Push_Pulled_Object":0.15,
			"Bike":0.1,
			"Door":0.1,
			"Construction_Barrier":0.05,
			"Dumpster":0.05,
			"Parking_Meter":0.05,
		}
		assert sum(eval_target_weight.values()) == 1.0

	if config.add_act:
		act_eval_target = ["vehicle_turning_right","vehicle_turning_left","Unloading","Transport_HeavyCarry","Opening","Open_Trunk","Loading","Exiting","Entering","Closing_Trunk","Closing","Interacts","Pull","Riding","Talking","activity_carrying","specialized_talking_phone","specialized_texting_phone"] # "vehicle_u_turn" is not used since not exists in val set
		act_eval_target = {one:1 for one in act_eval_target}
		act_eval_target_weight ={one:1.0/len(act_eval_target) for one in act_eval_target}
	
	self_summary_strs = Summary()
	stats = [] # tuples with {"metrics":,"step":,}
	# load the frame count data first
	
	train_data = read_data_diva(config,config.trainlst,config.imgpath,config.annopath,tococo=config.tococo) # True to filter data
	val_data = read_data_diva(config,config.vallst,config.valframepath,config.valannopath,tococo=False)#,randp=0.02)
	config.train_num_examples = train_data.num_examples

	# the total step (iteration) the model will run
	num_steps = int(math.ceil(train_data.num_examples/float(config.im_batch_size)))*config.num_epochs
	num_val_steps = int(math.ceil(val_data.num_examples/float(config.im_batch_size)))*1
	
	#config_vars = vars(config)
	#self_summary_strs.add("\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config_vars]))

	# two model, this is the lazy way
	#model = get_model(config) # input is image paths
	models = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models.append(get_model(config,i,controller=config.controller))

	config.is_train=False
	models_eval = []
	for i in xrange(config.gpuid_start, config.gpuid_start+config.gpu):
		models_eval.append(get_model(config,i,controller=config.controller))
	config.is_train=True
	
	trainer = Trainer(models,config)
	tester = Tester(models_eval,config,add_mask=config.add_mask) # need final box and stuff?

	saver = tf.train.Saver(max_to_keep=5) # how many model to keep
	bestsaver = tf.train.Saver(max_to_keep=5) # just for saving the best model

	# start training!
	# allow_soft_placement :  tf will auto select other device if the tf.device(*) not available

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
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

		loss = -1

		#best_ar = (-1.0,1)# (AR,best_step)
		best = (-1.0,1) # (average_ar + average_ap)/2
		for batch in tqdm(train_data.get_batches(config.im_batch_size,num_batches=num_steps),total=num_steps,ascii=True,smoothing=1):

			global_step = sess.run(models[0].global_step) + 1 # start from 0 or the previous step
			
			validation_performance = None
			if (global_step % config.save_period == 0) or (config.load and isStart and (config.ignore_vars is None)): # time to save model

				tqdm.write("step:%s/%s (epoch:%.3f), this(last) step loss:%.6f"%(global_step,num_steps,(config.num_epochs*global_step/float(num_steps)),loss))
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
						for val_batch in val_batches:
							# load the image here and resize
							image = cv2.imread(val_batch.data['imgs'][0],cv2.IMREAD_COLOR)
							imgid = os.path.splitext(os.path.basename(val_batch.data['imgs'][0]))[0]
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


						#boxes,probs,labels,_ = tester.step(sess,val_batch_)
						outputs = tester.step(sess,val_batch_)

						# post process this batch, also remember the ground truth
						for i in xrange(this_batch_num): # num gpu

							if config.add_act:
								boxes,labels,probs,actboxes,actlabels,actprobs = outputs[i]
							else:
								if config.add_mask:
									boxes,labels,probs,masks = outputs[i]
								else:
									boxes,labels,probs = outputs[i]
							scale = scales[i]
							val_batch = val_batches[i]

							boxes = boxes / scale

						
							target_dt_boxes = {one:[] for one in eval_target.keys()}
							for box, prob, label in zip(boxes,probs,labels):
								# coco box
								box[2] -= box[0]
								box[3] -= box[1]

								assert label > 0

								if config.tococo:
									cat_name = config.class_names[label]
								else:
									# diva class trained from scratch
									cat_name = targetid2class[label]

								target_class = None

								if config.tococo:
									for t in eval_target:
										if cat_name in eval_target[t]:
											target_class = t
								else:
									if cat_name in eval_target:
										target_class = cat_name

								if target_class is None: # box from other class of mscoco/diva
									continue

								prob = float(round(prob,2))
								box = list(map(lambda x:float(round(x,1)),box))

								target_dt_boxes[target_class].append((box,prob))
							#gt
							gt_boxes = {one:[] for one in eval_target.keys()}
							anno = val_batch.data['gt'][0]
							#print anno
							#print boxes,probs,labels
							#sys.exit()
							for box, label in zip(anno['boxes'],anno['labels']):
								label = targetid2class[label]
								if label in eval_target:
									gt_box = list(map(lambda x:float(round(x,1)),box))
									# gt_box is in (x1,y1,x2,y2)
									# convert to coco box
									gt_box[2]-=gt_box[0]
									gt_box[3]-=gt_box[1]

									gt_boxes[label].append(gt_box)
							# gt_boxes and target_dt_boxes for this image

							# eval on one single image
							for target_class in eval_target.keys():
								#if len(gt_boxes[target_class]) == 0:
								#	continue
								target_dt_boxes[target_class].sort(key=operator.itemgetter(1),reverse=True)
								d = [box for box,prob in target_dt_boxes[target_class]]
								dscores = [prob for box,prob in target_dt_boxes[target_class]]
								g = gt_boxes[target_class]

								dm,gm = match_detection(d,g,cocomask.iou(d,g,[0 for _ in xrange(len(g))]),iou_thres=0.5)

								e[target_class][imgid] = {
									"dscores":[],
									"dm":[],
									"gt_num":len(g),
								}

								
								e[target_class][imgid]['dscores'] = dscores
								e[target_class][imgid]['dm'] = dm

							# eval the act box as well, put stuff in e_act
							if config.add_act:
								target_act_dt_boxes = {one:[] for one in act_eval_target.keys()}
								for box, prob, label in zip(actboxes,actprobs,actlabels):
									# coco box
									box[2] -= box[0]
									box[3] -= box[1]

									assert label > 0					
									# diva class trained from scratch
									cat_name = targetactid2class[label]

									target_class = None
									
									if cat_name in act_eval_target:
										target_class = cat_name

									if target_class is None: # box from other class of mscoco/diva
										continue
									

									prob = float(round(prob,2))
									box = list(map(lambda x:float(round(x,1)),box))

									target_act_dt_boxes[target_class].append((box,prob))

								#gt
								gt_act_boxes = {one:[] for one in act_eval_target.keys()}
								anno = val_batch.data['gt'][0]
								#print anno
								#print boxes,probs,labels
								#sys.exit()
								for box, label in zip(anno['actboxes'],anno['actlabels']):
									label = targetactid2class[label]
									if label in act_eval_target:
										gt_box = list(map(lambda x:float(round(x,1)),box))
										# gt_box is in (x1,y1,x2,y2)
										# convert to coco box
										gt_box[2]-=gt_box[0]
										gt_box[3]-=gt_box[1]

										gt_act_boxes[label].append(gt_box)
								# gt_boxes and target_dt_boxes for this image

								# eval on one single image
								for target_class in act_eval_target.keys():
									#if len(gt_boxes[target_class]) == 0:
									#	continue
									target_act_dt_boxes[target_class].sort(key=operator.itemgetter(1),reverse=True)
									d = [box for box,prob in target_act_dt_boxes[target_class]]
									dscores = [prob for box,prob in target_act_dt_boxes[target_class]]
									g = gt_act_boxes[target_class]

									dm,gm = match_detection(d,g,cocomask.iou(d,g,[0 for _ in xrange(len(g))]),iou_thres=0.5)

									e_act[target_class][imgid] = {
										"dscores":[],
										"dm":[],
										"gt_num":len(g),
									}

									
									e_act[target_class][imgid]['dscores'] = dscores
									e_act[target_class][imgid]['dm'] = dm

							

					# we have the dm and g matching for each image in e & e_act
					# max detection per image per category
					aps,ars = aggregate_eval(e,maxDet=100)

					aps_str = "|".join(["%s:%.5f"%(class_,aps[class_]) for class_ in aps])
					ars_str = "|".join(["%s:%.5f"%(class_,ars[class_]) for class_ in ars])
					#tqdm.write("\tval in %s at step %s, AP:%s, AR:%s, previous best AR for %s at %s is %.5f"%(num_val_steps,global_step,aps_str,ars_str,eval_best,best[1],best[0]))
					#validation_performance = ars[eval_best]
					# now we use average AR and average AP or weighted
					average_ap,average_ar = weighted_average(aps,ars,eval_target_weight)


					ap_weight = 0.4
					ar_weight = 0.6
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

						act_perf_weight = 0.8
						obj_perf_weight = 0.2
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
				
			# skip if the batch is not complete, usually the last few ones
			if len(batch[1]) != config.gpu:
				continue

			loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss,train_op,act_losses = trainer.step(sess,batch)
			
			if math.isnan(loss):
				tqdm.write("warning, nan loss: loss:%s,rpn_label_loss:%s, rpn_box_loss:%s, fastrcnn_label_loss:%s, fastrcnn_box_loss:%s"%(loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss))
				if config.add_act:
					tqdm.write("\tact_losses:%s"%(act_losses))
				print "batch:%s"%(batch[1][0].data['imgs'])
				sys.exit()

			# save these for ploting later
			stats.append({
				"s":global_step,
				"l":loss,
				"val":validation_performance
			})

		# save the last model
		if global_step % config.save_period != 0: # time to save model
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

	all_images = [line.strip() for line in open(config.imgpath,"r").readlines()]

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

	if config.add_mask:
		# [R,14,14]
		model_final_masks = [model.final_masks for model in models]

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
			feed_dict = {}
			for i,image in enumerate(images):
				im = cv2.imread(image,cv2.IMREAD_COLOR)
				imagename = os.path.splitext(os.path.basename(image))[0]

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

			if config.add_mask:
				for _,boxes,labels,probs,masks in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs,model_final_masks):
					sess_input+=[boxes,labels,probs,masks]
			else:	
				for _,boxes,labels,probs in zip(range(len(images)),model_final_boxes,model_final_labels,model_final_probs):
					sess_input+=[boxes,labels,probs]

			outputs = sess.run(sess_input,feed_dict=feed_dict)
			if config.add_mask:
				pn = 4
			else:
				pn = 3
			outputs = [outputs[i*pn:(i*pn+pn)] for i in xrange(len(images))]


				

			for i,output in enumerate(outputs):
				scale = scales[i]
				ori_shape = ori_shapes[i]
				if config.add_mask:
					final_boxes, final_labels, final_probs, final_masks = output
					final_boxes = final_boxes / scale
					final_masks = [fill_full_mask(box,mask,ori_shape) for box,mask in zip(final_boxes,final_masks)]
				else:
					final_boxes, final_labels, final_probs = output
					final_boxes = final_boxes / scale
					final_masks = [None for one in final_boxes]	

				pred = []

				for box, prob, label, mask in zip(final_boxes,final_probs,final_labels,final_masks):
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
						"segmentation":rle
					}
					pred.append(res)

				# save the data
				resultfile = os.path.join(config.outbasepath,"%s.json"%imagename)
				with open(resultfile,"w") as f:
					json.dump(pred,f)





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
	for i in xrange(config.gpu):
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
		restore_vars = allvars
		opts = ["Adam","beta1_power","beta2_power","Adam_1","Adadelta_1","Adadelta","Momentum"]
		restore_vars = [var for var in restore_vars if var.name.split(":")[0].split("/")[-1] not in opts]
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
		else:
			raise Exception("mode %s not supported"%(config.mode))
