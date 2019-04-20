# coding=utf-8
# given original object annotation, generate new mix-up annotation. pair-wise
import os, sys, argparse, copy
import numpy as np
from tqdm import tqdm
import random
import pycocotools.mask as cocomask # for computing ious of bounding boxes
from check_small_obj_person_overlap import min_outer, max_outer

parser = argparse.ArgumentParser()
parser.add_argument("annopath")
parser.add_argument("framelst")
parser.add_argument("newannopath")

# 1. will only mix up frames from the same scene
# 2. same video?
parser.add_argument("--only_same_video", action="store_true", help="only mix frames from the same video")
parser.add_argument("--framegap", type=int, default=100, help="the two frame has to larger than this gap if in the same video")
# 3. allow bounding box overlap?
parser.add_argument("--allow_bbox_overlap", action="store_true", help="otherwise we check the box overlapping all classes in the only list")

parser.add_argument("--max_per_frame", type=int, default=10, help="maximum mixup frame for each original frame")

# only these classes would be considered adding in mix-up
only = [
	'Person',
	'Construction_Vehicle',
	'Push_Pulled_Object',
	'Bike',
	'Prop',
	'Skateboard',
	'Prop_Overshoulder',
	'Bike_Person',
	'Prop_Person',
	'Skateboard_Person',
	'Prop_Oversholder_Person'
]

def get_scene(videoname):
	s = videoname.split("_S_")[-1]
	s = s.split("_")[0]
	return s[:4]

if __name__ == "__main__":
	args = parser.parse_args()

	frames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(args.framelst).readlines()]

	# randomize so that we have a even mix up
	random.shuffle(frames)
	random.shuffle(frames)

	if not os.path.exists(args.newannopath):
		os.makedirs(args.newannopath)

	# n^2 ops. find mix-up for each frame
	for a_framename in tqdm(frames, ascii=True):
		a_scene = get_scene(a_framename)
		a_videoname = a_framename.split("_F_")[0]
		a_framenum = int(a_framename.split("_F_")[-1])

		a_anno_path = os.path.join(args.annopath, "%s.npz"%a_framename)
		if not os.path.exists(a_anno_path):
			continue
		a_anno = dict(np.load(a_anno_path))

		a_mix_count = 0

		valid = np.isin(a_anno['labels'], only)
		a_box_for_check = a_anno['boxes'][valid]
		a_label_for_check = a_anno['labels'][valid]

		for b_framename in frames:
			b_scene = get_scene(b_framename)
			b_videoname = b_framename.split("_F_")[0]
			# 1. needs to be the same scene
			if a_scene != b_scene:
				continue
			# 2. same video?
			if args.only_same_video and (a_videoname != b_videoname):
				continue

			b_framenum = int(b_framename.split("_F_")[-1])
			# 2.2 check frame gap
			if abs(a_framenum - b_framenum) <= args.framegap:
				continue

			# mix annotations
			b_anno_path = os.path.join(args.annopath, "%s.npz"%b_framename)
			if not os.path.exists(b_anno_path):
				continue
			b_anno = dict(np.load(b_anno_path))

			# filter out classes not considered in mixup
			valid = np.isin(b_anno['labels'], only)
			b_anno['boxes'] = b_anno['boxes'][valid]
			b_anno['labels'] = b_anno['labels'][valid]

			mixup_boxes = b_anno['boxes']
			mixup_labels = b_anno['labels']
			if len(mixup_boxes) == 0:
				continue

			if not args.allow_bbox_overlap:
				if len(a_box_for_check) != 0: # so all b_boxes could be used
					w = min_outer(a_box_for_check[:, 2], b_anno['boxes'][:, 2]) - max_outer(a_box_for_check[:, 0], b_anno['boxes'][:, 0])
					h = min_outer(a_box_for_check[:, 3], b_anno['boxes'][:, 3]) - max_outer(a_box_for_check[:, 1], b_anno['boxes'][:, 1])
					w[w < 0] = 0
					h[h < 0] = 0 

					intersection = w * h # [A, B]
					max_b_inter = np.amax(intersection, axis=0) # [B]

					b_valid = max_b_inter == 0.0

					mixup_boxes = b_anno['boxes'][b_valid]
					mixup_labels = b_anno['labels'][b_valid]
				
			
			if len(mixup_boxes) == 0:
				continue

			new_anno = copy.deepcopy(a_anno)
			new_anno['mixup_boxes'] = mixup_boxes
			new_anno['mixup_labels'] = mixup_labels

			new_anno_path = os.path.join(args.newannopath, "%s_M_%s.npz"%(a_framename, b_framename))
			np.savez_compressed(new_anno_path, **new_anno)

			a_mix_count+=1
			if a_mix_count >= args.max_per_frame:
				break





