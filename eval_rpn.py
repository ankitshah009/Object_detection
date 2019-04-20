# coding=utf-8
# given the RPN output, compute the performance

import sys,os,argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from class_ids import targetClass2id, targetClass2id_mergeProp
from utils import gather_gt, match_detection
import pycocotools.mask as cocomask

parser = argparse.ArgumentParser()
parser.add_argument("gtpath")
parser.add_argument("idlst")
parser.add_argument("outpath")
parser.add_argument("--merge_prop", action="store_true")


def tostr(list_):
	return [str(i) for i in list_]

def main(args):
	global targetClass2id
	videonames = [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(args.idlst).readlines()]

	if args.merge_prop:
		targetClass2id = targetClass2id_mergeProp


	dont_care = ["BG", "Tree", "Trees", "Articulated_Infrastructure", "Other"]
	for d in dont_care:
		del targetClass2id[d]

	targetid2class = {targetClass2id[one]:one for one in targetClass2id}

	topks = [30, 100, 500, 1000]
	ious = [0.3, 0.5]
	#topks=[1000]
	#ious=[0.5]
	# each topk's each class's recall on each image. each item is a tuple (matched, total_gt)
	recalls = {
		topk:{
			iou:{
				classname:[] for classname in targetClass2id} for iou in ious} for topk in topks}

	# load all the ground truth
	gt_not = 0
	noanno = 0
	out_not = 0
	for videoname in tqdm(videonames, ascii=True):
		gtfile = os.path.join(args.gtpath, "%s.npz" % videoname)
		if not os.path.exists(gtfile):
			gt_not+=1
			continue
		outfile = os.path.join(args.outpath, "%s.npy" % videoname)
		if not os.path.exists(outfile):
			out_not+=1
			continue

		gt = dict(np.load(gtfile)) # 'boxes' -> [K,4]  # 'labels' is str

		# convert the label str to label id, keep only the box that we care
		labels = []
		boxes = []
		for i, classname in enumerate(list(gt['labels'])):
			if targetClass2id.has_key(classname):
				labels.append(targetClass2id[classname])
				boxes.append(gt['boxes'][i])

		if not labels: # no anno for this frame
			noanno+=1
			continue

		boxes = np.array(boxes, dtype="float32")
		labels = np.array(labels)
		# put the ground truth into classname -> boxes
		# classname with no gt will be empty list
		# box is in (x, y, w, h)
		gt = gather_gt(boxes, labels, targetClass2id, targetid2class)

		# load the rpn output
		try:
			rpn_out = np.load(outfile)  # [K, 5], :4 is box, last is score
		except Exception as e:
			print "error loading %s, skipped..." % videoname
			continue
		rpn_boxes = rpn_out[:, :4]
		# to coco box
		rpn_boxes[:, 2] = rpn_boxes[:, 2] -  rpn_boxes[:, 0]
		rpn_boxes[:, 3] = rpn_boxes[:, 3] -  rpn_boxes[:, 1]

		rpn_scores = rpn_out[:,4]
		# sort first
		sorts = np.argsort(rpn_scores)[::-1]
		rpn_boxes = rpn_boxes[sorts]  # [K, 4]
		rpn_scores = rpn_scores[sorts]  # [K]

		# get the match for each class
		for topk in topks:
			for iou in ious:
				for classname in gt:
					matched = 0
					gt_num = len(gt[classname])

					g = gt[classname]
					d = rpn_boxes[:topk]
					if gt_num != 0:
						# [D, G]
						# -1 means not match to anything
						dm, gm = match_detection(d, g, cocomask.iou(d, g, [0 for _ in xrange(len(g))]), iou_thres=iou)

						matched = len([i for i in gm if i != -1])

					recalls[topk][iou][classname].append((matched, gt_num))


	# print out the results
	print "total id lst %s, gt file not exists %s, out file not exists %s, empty anno file %s" % (len(videonames), gt_not, out_not, noanno)

	for classname in targetClass2id:
		for topk in topks:
			for iou in ious:	
				total_gt = sum([gt_num for matched, gt_num in recalls[topk][iou][classname]])

				total_matched = sum([matched for matched, gt_num in recalls[topk][iou][classname]])
				if total_gt != 0:
					recall = total_matched/float(total_gt)
				else:
					recall = 0.0
				num_not_empty_gt = len([1 for matched, gt_num in recalls[topk][iou][classname] if gt_num != 0])

				print ",".join(tostr([topk, iou, classname, recall, num_not_empty_gt, len(recalls[topk][iou][classname])]))


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)