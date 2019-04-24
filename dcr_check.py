# coding=utf-8
# sanity check, use the dcr anno box, add nms, see the json performance
import sys, os, argparse, json
import numpy as np
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("npzpath")
parser.add_argument("jsonpath")

from class_ids import targetClass2id_new_nopo
targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id_new_nopo[one]:one for one in targetClass2id_new_nopo}


from utils import nms_wrapper

if __name__ == "__main__":
	args = parser.parse_args()

	npzfiles = glob(os.path.join(args.npzpath, "*npz"))

	if not os.path.exists(args.jsonpath):
		os.makedirs(args.jsonpath)

	box_nums = []
	result_nums = []
	for npzfile in tqdm(npzfiles, ascii=True):
		
		box_data = dict(np.load(npzfile))

		# [C, K]
		box_probs = box_data['frcnn_probs']

		# [K, 4]
		boxes = box_data['frcnn_boxes']

		C = box_probs.shape[0]
		assert C == len(targetClass2id) - 1 # - BG class

		K = boxes.shape[0]
		box_nums.append(K)

		# [K, 4] -> [C, K, 4]
		boxes = np.tile(np.expand_dims(boxes, axis=0), [C, 1, 1])

		args.result_score_thres = 0.0001
		args.fastrcnn_nms_iou_thres = 0.5
		args.result_per_im = 100
		final_boxes, final_labels, final_probs = nms_wrapper(boxes, box_probs, args)

		pred = []

		for j,(box, prob, label) in enumerate(zip(final_boxes, final_probs, final_labels)):
			box[2] -= box[0]
			box[3] -= box[1] # produce x, y, w, h output

			cat_id = int(label)
			cat_name = targetid2class[cat_id]
			
			rle = None
			
			res = {
				"category_id": cat_id,
				"cat_name": cat_name, # [0-80]
				"score": float(round(prob, 4)),
				"bbox": list(map(lambda x:float(round(x,1)),box)),
				"segmentation":rle,
			}

			pred.append(res)
		result_nums.append(len(pred))

		# save the data
		filename = os.path.splitext(os.path.basename(npzfile))[0]
		resultfile = os.path.join(args.jsonpath, "%s.json"%filename)
		with open(resultfile, "w") as f:
			json.dump(pred, f)
	print "box num median %s, max %s, min %s; result box num median %s, max %s, min %s"%(np.median(box_nums), max(box_nums), min(box_nums), np.median(result_nums), max(result_nums), min(result_nums))

