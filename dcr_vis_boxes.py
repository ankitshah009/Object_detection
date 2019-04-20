# coding=utf-8
# visualize the outputed boxes

import sys, os, argparse, operator, cv2
from utils import draw_boxes
import numpy as np
from glob import glob
from tqdm import tqdm
from class_ids import targetClass2id_new_nopo
targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]:one for one in targetClass2id}
"""
data = {
	"rcnn_boxes": rcnn_boxes, # [K, 4]
	"frcnn_boxes": final_boxes, # [C, K, 4]
	"frcnn_probs": final_probs, # [C, K] # C is num_class -1
	"det_labels": # [K], int
	"det_gt_max_ious": [K], float
}
"""
parser = argparse.ArgumentParser()
parser.add_argument("videoname")
parser.add_argument("npzpath", help="path to one npz per frame")
parser.add_argument("framepath")
parser.add_argument("outpath")
parser.add_argument("--topk", default=50, type=int, help="showing topk boxes per image")
parser.add_argument("--show", default="rpn", help="rpn|frcnn")

if __name__ == "__main__":
	args = parser.parse_args()

	npzfiles = glob(os.path.join(args.npzpath, "%s_F_*.npz"%args.videoname))

	npzfiles.sort()

	if not os.path.exists(args.outpath):
		os.makedirs(args.outpath)

	for npzfile in tqdm(npzfiles, ascii=True):
		framename = os.path.splitext(os.path.basename(npzfile))[0]
		framefile = os.path.join(args.framepath, args.videoname, framename+".jpg")

		data = np.load(npzfile)

		# TODO: change this to show other boxes

		# step 1, rank the boxes since we cant show all
		probs = np.amax(data['frcnn_probs'], axis=0) # [C, k] -> [K]
		frcnn_labels = np.argmax(data['frcnn_probs'], axis=0) # [K]
		if args.show =="rpn":
			boxes = data['rcnn_boxes'] # [K, 4]
		elif args.show == "frcnn":
			boxes = []
			for i in xrange(len(probs)):
				boxes.append(data['frcnn_boxes'][frcnn_labels[i], i, :])
			boxes = np.array(boxes)
		
		gt_labels = data['det_labels']

		box_and_other = [(boxes[i], probs[i], frcnn_labels[i], gt_labels[i]) for i in xrange(len(boxes))]

		box_and_other.sort(key=operator.itemgetter(1), reverse=True)
		box_and_other = box_and_other[:args.topk]

		boxes = np.array([b[0] for b in box_and_other], dtype="int")
		labels = ["%.3f | %s | %s" % (b[1], targetid2class[b[2]+1], targetid2class[b[3]]) for b in box_and_other]
		color = (255, 0, 0)
		ori_im = cv2.imread(framefile, cv2.IMREAD_COLOR)

		new_im = draw_boxes(ori_im, boxes, labels, color)

		target_img = os.path.join(args.outpath, "%s.jpg"%framename)
		cv2.imwrite(target_img, new_im)