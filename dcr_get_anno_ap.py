# coding=utf-8
# compute the AP for the faster rcnn classification given the annotation

import sys, os, argparse
import numpy as np
from dcr_models import compute_AP
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("annopath")
parser.add_argument("--skip", type=int, default=30, help='has to skip, since no memory will fit this')

from class_ids import targetClass2id_new_nopo
targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id_new_nopo[one]:one for one in targetClass2id_new_nopo}
eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object", "Bike"] #"Construction_Vehicle"]
eval_target = {one:1 for one in eval_target}

if __name__ == "__main__":
	args = parser.parse_args()
	# the annotation already include labels

	files = glob(os.path.join(args.annopath, "*.npz"))
	files.sort()
	files = files[::args.skip]

	e = {one:[] for one in eval_target}
	label_num = {one:0 for one in targetid2class} # the percentage (/num_boxes) for each class on ground truth labels
	total_box = 0
	for file in tqdm(files, ascii=True):
		data = dict(np.load(file))

		# [num_class-1, K]
		frcnn_probs = data['frcnn_probs']
		# [K], each is 0 -> num_class int
		det_labels = data['det_labels']
		assert len(det_labels) == frcnn_probs.shape[-1]

		for eval_class in e:
			classIdx = targetClass2id[eval_class]
			# [K]
			bin_labels = det_labels == classIdx
			# [K]
			this_probs = frcnn_probs[classIdx-1, :]
			e[eval_class].extend(zip(this_probs, bin_labels))
		for label in det_labels:
			label_num[label]+=1
			total_box+=1

	aps = []
	for eval_class in e:
		AP = compute_AP(e[eval_class])
		aps.append((eval_class, AP))
	average_ap = np.mean([ap for _, ap in aps])

	details = "|".join(["%s:%.5f"%(classname, ap) for classname, ap in aps])

	print average_ap
	print details
	print "total gt box %s, gt box distribution: %s" % (
		total_box,
		", ".join(["%s:%.6f"%(targetid2class[one], label_num[one]/float(total_box)) for one in label_num])
	)