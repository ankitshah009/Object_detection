# coding=utf-8
# given a file lst, ground truth and detection output, get the eval result

import sys,os,argparse,json
from tqdm import tqdm
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("filelst")
	parser.add_argument("gtpath")
	parser.add_argument("outpath")
	parser.add_argument("--not_coco_box", action="store_true")
	parser.add_argument("--merge_prop", action="store_true",help="this means put all Push_Pulled_Object anno into prop")
	parser.add_argument("--skip",type=int,default=1)
	parser.add_argument("--skip_not_exist_out", action="store_true")

	parser.add_argument("--limit", type=int, default=None, help="limit top k per json")
	return parser.parse_args()

def gather_dt(boxes, probs, labels,eval_target,not_coco_box=False):
	target_dt_boxes = {one:[] for one in eval_target}
	for box, prob, label in zip(boxes,probs,labels):

		assert label > 0
		if not_coco_box:
			box[2] -= box[0]
			box[3] -= box[1]

		target_class = None

		
		if label in eval_target:
			target_class = label

		if target_class is None: # box from other class of mscoco/diva
			continue

		prob = float(round(prob,4))
		box = list(map(lambda x:float(round(x,2)),box))

		target_dt_boxes[target_class].append((box,prob))
	return target_dt_boxes

def gather_gt(anno_boxes,anno_labels,eval_target):
	gt_boxes = {one:[] for one in eval_target}
	for box, label in zip(anno_boxes,anno_labels):
		if label in eval_target:
			gt_box = list(map(lambda x:float(round(x,1)),box))
			# gt_box is in (x1,y1,x2,y2)
			# convert to coco box
			gt_box[2]-=gt_box[0]
			gt_box[3]-=gt_box[1]

			gt_boxes[label].append(gt_box)
	return gt_boxes

from utils import match_dt_gt, aggregate_eval

if __name__ == "__main__":
	args = get_args()

	files = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(args.filelst,"r").readlines()]
	files.sort()
	files = files[::args.skip]

	# previous classes before annotation refining
	#eval_target = ["Vehicle","Person","Construction_Barrier","Construction_Vehicle", "Door","Dumpster","Prop","Push_Pulled_Object","Bike","Parking_Meter", "Prop_plus_Push_Pulled_Object"]
	eval_target = [
		"Vehicle",
		"Person",
		"Construction_Barrier",
		"Construction_Vehicle", 
		"Door",
		"Dumpster",
		"Prop",
		"Push_Pulled_Object",
		"Bike",
		"Parking_Meter",
		"Skateboard",
		"Prop_Overshoulder",
	]


	eval_target = {one:1 for one in eval_target}

	e = {one:{} for one in eval_target} # cat_id -> imgid -> {"dm","dscores"}

	count_no_out=0
	
	for filename in tqdm(files, ascii=True):
		gtfile = os.path.join(args.gtpath,"%s.npz"%filename)
		outfile = os.path.join(args.outpath,"%s.json"%filename)


		# load annotation first
		if not os.path.exists(gtfile):
			continue
		anno = dict(np.load(gtfile))

		if not os.path.exists(outfile):
			count_no_out+=1
			out = []
			if args.skip_not_exist_out:
				continue
		else:
			with open(outfile, "r") as f:
				out = json.load(f)

		if args.merge_prop:
			
			for i,one in enumerate(out):
				if one['cat_name'] == "Push_Pulled_Object" or one['cat_name'] == "Prop":
					out[i]['cat_name'] = "Prop_plus_Push_Pulled_Object"

			# change ground truth, too # groung truth already has it # v1-validate_actgt_allsingle_mergeprop_npz
			#for i,one in enumerate(anno['labels']):
			#	if one == "Push_Pulled_Object" or one == "Prop":
			#		anno['labels'][i] = "Prop_plus_Push_Pulled_Object"

		if args.limit is not None:
			out.sort(key=operator.itemgetter("score"), reverse=True)
			out = out[:args.limit]
		boxes = [one['bbox'] for one in out]
		probs = [one['score'] for one in out]	
		labels = [one['cat_name'] for one in out]

		target_dt_boxes = gather_dt(boxes,probs,labels,eval_target,not_coco_box=args.not_coco_box)

		gt_boxes = gather_gt(anno['boxes'],anno['labels'],eval_target)

		match_dt_gt(e,filename,target_dt_boxes,gt_boxes,eval_target)

	print "%s/%s out file not exists"%(count_no_out, len(files))
	aps,ars = aggregate_eval(e, maxDet=100)
	aps_str = "|".join(["%s:%.5f"%(class_,aps[class_]) for class_ in aps])
	ars_str = "|".join(["%s:%.5f"%(class_,ars[class_]) for class_ in ars])
	classes = sorted(aps.keys())
	headers = ['metric'] + classes
	print ",".join(headers)
	print ",".join(["AP"] + ["%.6f"%aps[c] for c in classes])
	print ",".join(["AR"] + ["%.6f"%ars[c] for c in classes])





