# coding=utf-8
# given the anno in npz , get the small class box overlap rate with person boxes

import sys
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gtpath")
parser.add_argument("--check", default="Prop", type=str)
parser.add_argument("--scene", default=None, type=str)
parser.add_argument("--show_zero", action="store_true", help="show the zero rate box's image name and video names")


def min_outer(vec1, vec2):
	return np.transpose(np.minimum(vec1, np.transpose(np.expand_dims(vec2, axis=0))))

def max_outer(vec1, vec2):
	return np.transpose(np.maximum(vec1, np.transpose(np.expand_dims(vec2, axis=0))))

def get_scene(filename):
	videoname = os.path.splitext(os.path.basename(filename))[0].split("_F_")[0]
	s = videoname.split("_S_")[-1]
	s = s.split("_")[0]
	return s[:4]

if __name__ == "__main__":
	args = parser.parse_args()

	gtfiles = glob(os.path.join(args.gtpath,"*.npz"))

	inter_check_person_num, no_check_num, no_person_num = 0, 0, 0

	overlap_rates = []
	valid_img_num = 0

	zero_rate_framenames = []
	zero_rate_videonames = []
	for gtfile in tqdm(gtfiles, ascii=True):
		if args.scene is not None:
			scene = get_scene(gtfile)
			if args.scene != scene:
				continue
		valid_img_num+=1
		anno = dict(np.load(gtfile))

		check_boxes = []
		person_boxes = []

		for i,classname in enumerate(list(anno['labels'])):
			
			if classname == "Person":
				person_boxes.append(anno['boxes'][i])
			elif classname == args.check:
				check_boxes.append(anno['boxes'][i])

		if len(check_boxes) == 0:
			no_check_num += 1
			continue
		if len(person_boxes) == 0:
			no_person_num += 1
			continue
		inter_check_person_num += 1

		# boxes are x1,y1,x2,y2
		check_boxes = np.array(check_boxes,dtype="float32")  # [C, 4]
		person_boxes = np.array(person_boxes,dtype="float32")  # [P, 4]

		C, P = check_boxes.shape[0], person_boxes.shape[0]

		# get the intersection
		w = min_outer(check_boxes[:, 2], person_boxes[:, 2]) - max_outer(check_boxes[:, 0], person_boxes[:, 0]) 
		h = min_outer(check_boxes[:, 3], person_boxes[:, 3]) - max_outer(check_boxes[:, 1], person_boxes[:, 1])

		w[w < 0] = 0
		h[h < 0] = 0  # [C, P]

		intersection = w * h

		check_intersection = np.amax(intersection, axis=1) # [C] # each object box max intersection with the perosn box

		check_area = (check_boxes[:, 2] - check_boxes[:, 0]) * (check_boxes[:, 3] - check_boxes[:, 1])

		check_overlap_rate = check_intersection / check_area # [C]

		overlap_rates.extend(list(check_overlap_rate))

		if len([c for c in check_overlap_rate if c ==0]) > 0:
			zero_rate_framenames.append(os.path.splitext(os.path.basename(gtfile))[0])
			zero_rate_videonames.append(os.path.splitext(os.path.basename(gtfile))[0].split("_F_")[0])

	if args.show_zero:
		print "-"*40
		zero_rate_framenames.sort()
		for o in zero_rate_framenames:
			print o
		print "-"*40
		zero_rate_videonames = list(set(zero_rate_videonames))
		for o in zero_rate_videonames:
			print o

	print "checking %s, total imgs %s, no check %s, no person %s, valid %s"%(args.check, valid_img_num, no_check_num, no_person_num, inter_check_person_num)

	print "\toverlap: boxes %s, zero rate num %s, min overlap_rates %s, max %s, median %s, mean %s"%(len(overlap_rates), len([c for c in overlap_rates if c==0]), min(overlap_rates), max(overlap_rates), np.median(overlap_rates), np.mean(overlap_rates))


