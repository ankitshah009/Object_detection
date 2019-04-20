# coding=utf-8
# given object output path for each image, rerank prop or push pull object based on person box

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
	parser.add_argument("--factor", default=1.0, type=float)
	parser.add_argument("--person_thres", default=0.5, type=float)
	return parser.parse_args()


# return box and classname separately
def load_boxes(box_list,filters):
	scores = []
	boxes = []
	indices = []
	for i,one in enumerate(box_list):
		if one['cat_name'] in filters:
			boxes.append(one['bbox']) # x,y,w,h
			scores.append(one['score'])
			indices.append(i)
	return np.array(boxes), np.array(scores), indices

if __name__ == "__main__":
	args = get_args()

	if not os.path.exists(args.newpath):
		os.makedirs(args.newpath)

	objfiles = glob(os.path.join(args.outpath, "*.json"))

	filters = ["Prop", "Push_Pulled_Object", "Bike", "Prop_plus_Push_Pulled_Object"]


	for objfile in tqdm(objfiles, ascii=True):

		with open(objfile) as f:
			objboxes = json.load(f)

		# 1. get all the person boxes  higher conf
		# coco box
		person_boxes = np.array([obj['bbox'] for obj in objboxes if (obj['cat_name'] == "Person") and (obj['score'] >= args.person_thres)])
		person_scores = np.array([obj['score'] for obj in objboxes if (obj['cat_name'] == "Person") and (obj['score'] >= args.person_thres)])

		# 2. change the object's original score based on distance/iou with person box
		target_boxes, scores, indices = load_boxes(objboxes, filters)
		
		# change these boxes's score
		if len(person_boxes) != 0:
			ious = cocomask.iou(target_boxes, person_boxes, [0 for _ in xrange(len(target_boxes))])
			for i, box in enumerate(target_boxes):
				# [N, M], N is person boxes
				iou = ious[i] * person_scores # num_person_box
				assert len(iou) == len(person_boxes)

				scores[i] = scores[i] + args.factor * np.max(ious[i] * person_scores)


				
		# 3. save all the rest first
		objs = [obj for obj in objboxes if obj['cat_name'] not in filters]

		target_objs = []
		# replace the scores
		for i,idx in enumerate(indices):
			obj = objboxes[idx]
			obj['score'] = scores[i]
			target_objs.append(obj)

		objs.extend(target_objs)

		targetfile = os.path.join(args.newpath, "%s"%os.path.basename(objfile))

		with open(targetfile,"w") as f:
			json.dump(objs, f)

