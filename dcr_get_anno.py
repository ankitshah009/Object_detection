# coding=utf-8
# given the box output and annotation, get the box annotation based on iou
import sys, os, argparse
import numpy as np
import pycocotools.mask as cocomask
from tqdm import tqdm
from glob import glob
from class_ids import targetClass2id_new_nopo

parser = argparse.ArgumentParser()
parser.add_argument("boxpath")
parser.add_argument("annopath")
parser.add_argument("newpath")

if __name__ == "__main__":
	args = parser.parse_args()

	boxfiles = glob(os.path.join(args.boxpath, "*.npz"))

	targetClass2id = targetClass2id_new_nopo
	targetid2class = {targetClass2id_new_nopo[one]:one for one in targetClass2id_new_nopo}

	if not os.path.exists(args.newpath):
		os.makedirs(args.newpath)

	got_valid = 0
	for boxfile in tqdm(boxfiles, ascii=True):
		filename = os.path.splitext(os.path.basename(boxfile))[0]

		annofile = os.path.join(args.annopath, "%s.npz"%filename)
		if not os.path.exists(annofile):
			continue

		anno = dict(np.load(annofile))

		# get all relevant boxes first
		gt_boxes = []
		gt_labels = [] # int
		for i, classname in enumerate(list(anno['labels'])):
			if targetClass2id.has_key(classname):
				gt_boxes.append(anno['boxes'][i])
				gt_labels.append(targetClass2id[classname])

		if len(gt_boxes) == 0:
			continue

		gt_boxes = np.array(gt_boxes, dtype="float32") # [G, 4]

		dets = dict(np.load(boxfile))
		#det_boxes = dets['rcnn_boxes'].copy() # [K, 4]
		det_boxes = dets['frcnn_boxes'].copy() # [K, 4]

		# convert both to coco box format
		det_boxes[:, 2] = det_boxes[:, 2] - det_boxes[:, 0]
		det_boxes[:, 3] = det_boxes[:, 3] - det_boxes[:, 1]
		gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
		gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
		
		# get iou 
		# [K, G]
		ious = cocomask.iou(det_boxes, gt_boxes, [0 for _ in xrange(len(gt_boxes))])

		det_max_ious = np.amax(ious, axis=1) # [K]
		det_max_ious_wrt_gt = np.argmax(ious, axis=1) # [K]
		det_labels = []
		all_bg = True
		for i in xrange(len(det_max_ious_wrt_gt)):
			if det_max_ious[i] == 0.0: # TODO: set ious?
				cat = 0
			else:
				cat = gt_labels[det_max_ious_wrt_gt[i]]
				all_bg = False
			det_labels.append(cat)

		if all_bg:
			continue

		det_labels = np.array(det_labels, dtype="int")
		det_max_ious = np.array(det_max_ious, dtype="float")

		dets['det_labels'] = det_labels
		dets['det_gt_max_ious'] = det_max_ious

		target_file = os.path.join(args.newpath, "%s.npz"%filename)
		np.savez(target_file, **dets)
		got_valid+=1
	print "got valid frame %s, others discard due to zero gt or all detection box has no overlap with gt" % got_valid

