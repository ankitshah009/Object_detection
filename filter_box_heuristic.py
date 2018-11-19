# coding=utf-8
# given object output path for each image, filter prop or push pull object based on person box

import sys,os,argparse,json
from tqdm import tqdm
import pycocotools.mask as cocomask
#from viz import to_coco_box
from glob import glob
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("outpath")
	parser.add_argument("newpath")

	parser.add_argument("--iou_thres",default=0, type=float)
	parser.add_argument("--person_score_thres",default=0.7,type=float)
	parser.add_argument("--no_bike",action="store_true")
	return parser.parse_args()


# return box and classname separately
def load_boxes(box_list,filters):
	box_indexes = []
	boxes = []
	for i,one in enumerate(box_list):
		if one['cat_name'] in filters:
			boxes.append(one['bbox']) # x,y,w,h
			box_indexes.append(i)
	return np.array(boxes),box_indexes

if __name__ == "__main__":
	args = get_args()

	if not os.path.exists(args.newpath):
		os.makedirs(args.newpath)

	objfiles = glob(os.path.join(args.outpath, "*.json"))

	filters = ["Prop","Push_Pulled_Object","Bike"]
	if args.no_bike:
		filters = ["Prop","Push_Pulled_Object"]
	for objfile in tqdm(objfiles, ascii=True):

		with open(objfile) as f:
			objboxes = json.load(f)

		# 1. get all the person boxes  higher conf
		# coco box
		person_boxes = np.array([obj['bbox'] for obj in objboxes if (obj['cat_name'] == "Person") and (obj['score']>=args.person_score_thres)])

		# 2. get the prop and pull object ious
		filter_boxes,ori_indexes = load_boxes(objboxes,filters)
		if (len(filter_boxes) == 0) or (len(person_boxes) == 0):
			good_indexes = []
		else:

			# [N,M], N is person boxes
			ious = cocomask.iou(person_boxes,filter_boxes,[0 for _ in xrange(len(filter_boxes))])
			max_iou_forobj = np.max(ious, axis=0)# [M]
			good_indexes = [o for i,o in enumerate(ori_indexes) if max_iou_forobj[i] > args.iou_thres]

		# 3. save all the rest first
		objs = [obj for obj in objboxes if obj['cat_name'] not in filters]

		objs.extend([objboxes[i] for i in good_indexes])

		targetfile = os.path.join(args.newpath, "%s"%os.path.basename(objfile))

		with open(targetfile,"w") as f:
			json.dump(objs, f)

