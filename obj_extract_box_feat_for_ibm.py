# coding=utf-8
# given the ibm object detection in pickles, and the test video path, and our object model, get the feature for each boxes
# example:
# junweil@vid-gpu4:/mnt/sdc/junweil/object_detection/diva/ibm_detections$ python ~/object_detection/script/tf_mrcnn/obj_extract_box_feat_for_ibm.py small_test ibm_test1ab/ small_test.lst ../mrcnn101_fpn_newclass3_dilated/mrcnn101/01/save-best/ out_boxfeat --threshold_conf 0.9 --frame_gap 8

import sys,os,argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import cv2

from models import get_model_feat, resizeImage

import math, time, json, random, operator
import cPickle as pickle
import pycocotools.mask as cocomask


from utils import Dataset, Summary, get_op_tensor_name

# our model classname to classid
targetClass2id = { 
	"BG": 0,
	"Vehicle": 1,
	"Person": 2,
	"Parking_Meter": 3,
	"Tree": 4,
	"Skateboard": 5,
	"Prop_Overshoulder": 6,
	"Construction_Barrier": 7,
	"Door": 8,
	"Dumpster": 9,
	"Push_Pulled_Object": 10,
	"Construction_Vehicle": 11,
	"Prop": 12,
	"Bike": 13,
	"Animal": 14,
}

targetid2class = {targetClass2id[one]:one for one in targetClass2id}

ibm_idx_to_classname = {
	0: "Door",
	1: "Tree",
	3: "Dumpster",
	4: "Parking_Meter",
	7: "Construction_Barrier",
	#8: "Other",
	9: "Person",
	10: "Vehicle",
	11: "Bike",
	12: "Construction_Vehicle",
	13: "Push_Pulled_Object",
	14: "Prop",
	15: "Animal",
}
ibm_classname_to_idx = {v:k for k,v in ibm_idx_to_classname.iteritems()}

# only these class boxes are collected to extract features
classes_we_care = ['Person', 'Vehicle']

def get_args():
	global targetClass2id, targetid2class
	parser = argparse.ArgumentParser()
	parser.add_argument("video_path", help="path/to/video")
	parser.add_argument("ibm_box_path", help='ibm detection in pickle format')
	parser.add_argument("video_lst", help="a list of the videos with appendix")
	parser.add_argument("model_path", help="path to our obj detect model")
	parser.add_argument("out_path", help="save the feature to npz file for each frame, with boxes, classes and features")

	parser.add_argument("--frame_gap", default=8, type=int)
	parser.add_argument("--threshold_conf", default=0.0001, type=float, help="filtered ibm boxes using this")


	# ---- gpu params
	parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
	parser.add_argument("--gpuid_start", default=0, type=int, help="start of gpu id")
	parser.add_argument('--im_batch_size', type=int, default=1)
	parser.add_argument("--use_all_mem", action="store_true")

	# ----------- model params
	parser.add_argument("--num_class", type=int, default=15, help="num catagory + 1 background")
	parser.add_argument("--rpn_batch_size", type=int, default=256, help="num roi per image for RPN  training")
	parser.add_argument("--frcnn_batch_size", type=int, default=512, help="num roi per image for fastRCNN training")
	parser.add_argument("--rpn_test_post_nms_topk", type=int, default=1000 ,help="test post nms, input to fast rcnn")
	parser.add_argument("--max_size", type=int, default=1920, help="num roi per image for RPN and fastRCNN training")
	parser.add_argument("--short_edge_size", type=int, default=1080, help="num roi per image for RPN and fastRCNN training")

	parser.add_argument("--resnet152",action="store_true",help="")
	parser.add_argument("--resnet50",action="store_true",help="")
	parser.add_argument("--resnet34",action="store_true",help="")
	parser.add_argument("--resnet18",action="store_true",help="")
	parser.add_argument("--use_se",action="store_true",help="use squeeze and excitation in backbone")
	parser.add_argument("--use_frcnn_class_agnostic", action="store_true", help="use class agnostic fc head")
	parser.add_argument("--use_att_frcnn_head", action="store_true",help="use attention to sum [K, 7, 7, C] feature into [K, C]")


	# --------------- exp junk
	parser.add_argument("--use_dilations", action="store_true", help="use dilations=2 in res5")
	parser.add_argument("--use_deformable", action="store_true", help="use deformable conv")

	args = parser.parse_args()
	
	assert args.gpu == args.im_batch_size # one gpu one image

	args.controller = "/cpu:0" # parameter server

	targetid2class = targetid2class
	targetClass2id = targetClass2id

	assert len(targetClass2id) == args.num_class, (len(targetClass2id), args.num_class)

	# for ibm stuff
	args.ibm_classids_we_care = [ibm_classname_to_idx[classname] for classname in classes_we_care]

	# we will use these to convert the ibm class id to our targetclassid
	args.ibm_idx_to_classname = ibm_idx_to_classname
	args.targetClass2id = targetClass2id


	# ------- 04/2019 dilated
	args.use_dilations = True

	# ---------------more defautls
	args.add_act = False
	args.is_cascade_rcnn = False
	args.add_relation_nn = False
	args.diva_class3 = True
	args.diva_class = False
	args.diva_class2 = False
	args.use_small_object_head = False
	args.use_so_score_thres = False
	args.use_so_association = False
	args.use_gn = False
	args.so_person_topk = 10
	args.use_conv_frcnn_head = False
	args.use_cpu_nms = False
	args.use_bg_score = False
	args.freeze_rpn = True
	args.freeze_fastrcnn = True
	args.freeze = 2
	args.small_objects = ["Prop", "Push_Pulled_Object", "Prop_plus_Push_Pulled_Object", "Bike"]
	args.no_obj_detect = False
	args.add_mask = False
	args.is_fpn = True
	#args.new_tensorpack_model = True
	args.mrcnn_head_dim = 256
	args.is_train = False

	args.rpn_min_size = 0
	args.rpn_proposal_nms_thres = 0.7
	args.anchor_strides = (4, 8, 16, 32, 64)
		

	args.fpn_resolution_requirement = float(args.anchor_strides[3]) # [3] is 32, since we build FPN with r2,3,4,5?
		
	args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * args.fpn_resolution_requirement

	args.fpn_num_channel = 256

	args.fpn_frcnn_fc_head_dim = 1024

	# ---- all the mask rcnn config

	args.resnet_num_block = [3, 4, 23, 3] # resnet 101
	args.use_basic_block = False # for resnet-34 and resnet-18
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
	
	args.anchor_stride = 16 # has to be 16 to match the image feature total stride
	args.anchor_sizes = (32,64,128,256,512)

	args.anchor_ratios = (0.5, 1, 2)
	

	args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
	# iou thres to determine anchor label
	#args.positive_anchor_thres = 0.7
	#args.negative_anchor_thres = 0.3

	# when getting region proposal, avoid getting too large boxes
	args.bbox_decode_clip = np.log(args.max_size / 16.0)


	# fastrcnn
	args.fastrcnn_batch_per_im = args.frcnn_batch_size
	args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype='float32')
	
	args.fastrcnn_fg_thres = 0.5 # iou thres
	#args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

	# testing
	args.rpn_test_pre_nms_topk = 6000

	args.fastrcnn_nms_iou_thres = 0.5

	args.result_score_thres = args.threshold_conf
	args.result_per_im = 100 

	return args

def initialize(config, sess):
	tf.global_variables_initializer().run()
	allvars = tf.global_variables()
	allvars = [var for var in allvars if "global_step" not in var.name]
	restore_vars = allvars
	opts = ["Adam","beta1_power","beta2_power","Adam_1","Adadelta_1","Adadelta","Momentum"]
	restore_vars = [var for var in restore_vars if var.name.split(":")[0].split("/")[-1] not in opts]

	saver = tf.train.Saver(restore_vars, max_to_keep=5)

	load_from = config.model_path	
	ckpt = tf.train.get_checkpoint_state(load_from)
	if ckpt and ckpt.model_checkpoint_path:
		loadpath = ckpt.model_checkpoint_path					
		saver.restore(sess, loadpath)
	else:
		raise Exception("Model not exists")

# check argument
def check_args(args):
	print "cv2 version %s"%(cv2.__version__)

# return boxes [N, 4], labels, probs [N], 
def get_ibm_detections(ibm_data, frame_idx, threshold_conf, ibm_classids_we_care, ibm_idx_to_classname, targetClass2id):
	boxes = []
	labels = []
	probs = []
	for ibm_classid in ibm_classids_we_care:
		dets = ibm_data[ibm_classid][frame_idx] # [K, 5]
		our_classId = targetClass2id[ibm_idx_to_classname[ibm_classid]]
		for i in xrange(len(dets)):
			x1, y1, x2, y2, prob = dets[i]
			if prob <= threshold_conf:
				continue
			boxes.append([x1, y1, x2, y2])
			labels.append(our_classId)
			probs.append(prob)
	return np.array(boxes), np.array(labels), np.array(probs)

if __name__ == "__main__":
	args = get_args()

	check_args(args)

	videofiles = [os.path.join(args.video_path, one.strip()) for one in open(args.video_lst).readlines()]

	if not os.path.exists(args.out_path):
		os.makedirs(args.out_path)

	# 1. load the object detection model

	model = get_model_feat(args, args.gpuid_start, controller=args.controller)

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	if not args.use_all_mem:
		tfconfig.gpu_options.allow_growth = True
	tfconfig.gpu_options.visible_device_list = "%s"%(",".join(["%s"%i for i in range(args.gpuid_start, args.gpuid_start+args.gpu)]))

	with tf.Session(config=tfconfig) as sess:

		initialize(config=args,sess=sess)

		for videofile in tqdm(videofiles, ascii=True):
			# 2. read the video file
			try:
				vcap = cv2.VideoCapture(videofile)
				if not vcap.isOpened():
					raise Exception("cannot open %s"%videofile)
			except Exception as e:
				raise e

			videoname = os.path.splitext(os.path.basename(videofile))[0]
			video_out_path = os.path.join(args.out_path, videoname)
			if not os.path.exists(video_out_path):
				os.makedirs(video_out_path)

			# get ibm detection file, assume they are ${videoname}_detections.pkl
			ibm_file = os.path.join(args.ibm_box_path, "%s_detections.pkl" % videoname)
			assert os.path.exists(ibm_file), "ibm file %s not exists!"%ibm_file

			with open(ibm_file, "rb") as f:
				ibm_data = pickle.load(f)
					
			#frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
			#frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
			#fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
			# opencv 2
			#frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
			# opencv 3
			frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

			# check ibm frame count
			ibm_data_frame_count = None
			for classIdx in ibm_idx_to_classname:
				if ibm_data_frame_count is None:
					ibm_data_frame_count = len(ibm_data[classIdx])
				else:
					assert ibm_data_frame_count == len(ibm_data[classIdx])
			if ibm_data_frame_count != frame_count:
				tqdm.write("warning, %s ibm says %s frames but opencv says %s frames" % (videname, ibm_data_frame_count, frame_count))

			# we will only extract the minimum frames
			frame_count = min(frame_count, ibm_data_frame_count)

			# 3. read frame one by one
			cur_frame=0
			frame_stack = []
			while cur_frame < frame_count:
				suc, frame = vcap.read()
				if not suc:
					cur_frame+=1
					tqdm.write("warning, %s frame of %s failed"%(cur_frame, videoname))
					continue

				# skip some frame if frame_gap >1
				if cur_frame % args.frame_gap != 0:
					cur_frame+=1
					continue

				im = frame.astype("float32")
				h, w = im.shape[:2] # original width/height

				resized_image = resizeImage(im, args.short_edge_size, args.max_size)
				newh, neww = resized_image.shape[:2]

				scale = (newh*1.0/h + neww*1.0/w)/2.0

				# get ibm detection for this frame, only the objects we interested, and above confidence threshold
				boxes, labels, probs = get_ibm_detections(ibm_data, cur_frame, args.threshold_conf, args.ibm_classids_we_care, args.ibm_idx_to_classname, args.targetClass2id)

				# what if boxes is empty?

				# resize boxes # assuming the ibm boxes is in original im size
				# boxes # (x1,y1,x2,y2) # [K,4]
				boxes = boxes[:, [0, 2, 1, 3]] #(x1,x2,y1,y2)
				boxes = boxes.reshape((-1, 2, 2)) # (x1,x2),(y1,y2)
				boxes[:, 0] = boxes[:, 0] * (neww*1.0 / w) # x1,x2
				boxes[:, 1] = boxes[:, 1] * (newh*1.0 / h) # y1,y2
				boxes = boxes.reshape((-1, 4))
				boxes = boxes[:, [0, 2, 1, 3]] #(x1,y1,x2,y2)

				feed_dict = model.get_feed_dict(resized_image, boxes)

				
				sess_input = [model.final_box_features]

				# [N, 256]
				final_boxe_features, = sess.run(sess_input, feed_dict=feed_dict)
				assert len(final_boxe_features) == len(boxes)
					
				final_boxes = boxes / scale

				final_labels = labels
				final_probs = probs


				data = {
					'boxes': final_boxes,
					'labels': final_labels,
					'probs': final_probs,
					'box_features': final_boxe_features
				}
				
				predfile = os.path.join(video_out_path, "%s_F_%08d.npz"%(os.path.splitext(videoname)[0], cur_frame))
				np.savez(predfile, **data)				

				cur_frame+=1



