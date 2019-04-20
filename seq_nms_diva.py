# coding=utf-8
# given the object detection frame-level output, do seq-nms to generate new results
# https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/master/fgfa_rfcn/demo.py

import sys, os, argparse, json
from tqdm import tqdm
import numpy as np
from seq_nms import seq_nms2
from glob import glob
from nn import nms

parser = argparse.ArgumentParser()
parser.add_argument("videolst")
parser.add_argument("outpath")
parser.add_argument("newoutpath")
parser.add_argument("--job",type=int,default=1,help="total job")
parser.add_argument("--curJob",type=int,default=1,help="this script will execute job number")

classes = ( 
	"BG",
	"Vehicle",
	"Person",
	"Parking_Meter",
	"Tree",
	"Skateboard",
	"Prop_Overshoulder",
	"Construction_Barrier",
	"Door",
	"Dumpster",
	"Push_Pulled_Object",
	"Construction_Vehicle",
	"Prop",
	"Bike",
	"Animal",
)

# x1y1x2y2 to xywh
def to_coco_box(box):
	return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

if __name__ == "__main__":
	args = parser.parse_args()
	nms_iou_thres = 0.5

	videonames = [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(args.videolst).readlines()][::-1]

	if not os.path.exists(args.newoutpath):
		os.makedirs(args.newoutpath)
	count=0
	for videoname in tqdm(videonames, ascii=True):
		count+=1
		if((count % args.job) != (args.curJob-1)):
			continue
		# gather the frame-level annotation into one big list for each video
		frames = glob(os.path.join(args.outpath, "%s_F_*.json"%videoname))
		frames.sort()
		dets = [[] for i in classes[1:]] # except background
		for frame in frames:
			
			with open(frame, "r") as f:
				out = json.load(f)

			for cls_i, classname in enumerate(classes[1:]):
				
				boxes = np.array([one['bbox'] for one in out if one['cat_name'] == classname], dtype="float32")
				scores = np.array([one['score'] for one in out if one['cat_name'] == classname], dtype="float32")
				scores = np.expand_dims(scores, axis=1)
				# x,y,w,h -> x1,y1,x2,y2
				if len(boxes) > 0:
					boxes[:, 2] += boxes[:, 0]
					boxes[:, 3] += boxes[:, 1]
				else:
					boxes = np.zeros((0, 4), dtype="float")
				this_frame_this_class = np.hstack((boxes, scores)) # [num_box, 5]
				dets[cls_i].append(this_frame_this_class)

		# [num_frame, ...]
		#newboxes, newclasses, newscores = seq_nms(dets, classes)
		dets = seq_nms2(dets, classes)
		
		for i, frame in enumerate(frames):
			# do frame-level NMS again
			res = []
			"""
			for j in xrange(len(newboxes[i])):
				classIdx = newclasses[i][j]
				classname = classes[classIdx]
				bbox = to_coco_box(newboxes[i][j])
				score = newscores[i][j]
				res.append({
					"category_id":classIdx,
					"cat_name":classname, #[0-80]
					"score":float(round(score, 5)),
					"bbox": list(map(lambda x:float(round(x,1)),bbox)),
					"segmentation":None,
				})
			"""
			for c in xrange(len(dets)):
				this_dets = dets[c][i] # [N, 5]
				this_dets = this_dets.astype("float32")
				keep = nms(this_dets, nms_iou_thres)
				this_dets = this_dets[keep, :]
				classIdx = c + 1
				classname = classes[classIdx]
				for n in xrange(len(this_dets)):
					bbox = to_coco_box(this_dets[n, :4])
					score = this_dets[n, 4]
					res.append({
						"category_id":classIdx,
						"cat_name":classname, #[0-80]
						"score":float(round(score, 5)),
						"bbox": list(map(lambda x:float(round(x,1)),bbox)),
						"segmentation":None,
					})

			framename = os.path.splitext(os.path.basename(frame))[0]
			target_file = os.path.join(args.newoutpath, "%s.json"%framename)
			with open(target_file, "w") as f:
				json.dump(res, f)

