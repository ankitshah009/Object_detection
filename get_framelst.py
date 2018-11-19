# coding=utf-8
# given video lst and annopath, get frame lst

import sys,os,argparse,random

import numpy as np

from tqdm import tqdm
from glob import glob



def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("videolst")
	parser.add_argument("annopath")
	parser.add_argument("framelst")
	parser.add_argument("--skip",type=int,default=10)
	parser.add_argument("--check_frame",default=None)
	return parser.parse_args()

from class_ids import targetClass2id

if __name__ == "__main__":
	args = get_args()

	"""
	selects = {
		"Bike":0.7,
		"Push_Pulled_Object":0.3,
		"Prop":0.5,
		"Door":0.02
	}
	"""
	# select 2
	selects = {
		"Bike":1.0,
		"Push_Pulled_Object":0.3,
		"Prop":1.0,
		"Door":0.01
	}

	videos = [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(args.videolst).readlines()]

	# get the annotation stats for all frames
	label_dist = {classname:[] for classname in targetClass2id} # class -> [] num_box in each image
	label_dist_all = []
	framenames = []

	ignored_classes = {}
	for videoname in tqdm(videos,ascii=True):
		frames = glob(os.path.join(args.annopath, "%s_F_*.npz"%videoname))
		if args.check_frame is not None:
			newframes = []
			for f in frames:
				filename = os.path.splitext(os.path.basename(f))[0]
				framefile = os.path.join(args.check_frame, "%s"%videoname, "%s.jpg"%filename)
				if os.path.exists(framefile):
					newframes.append(f)
			if not len(newframes) == len(frames):
				tqdm.write("%s got %s/%s frames"%(videoname, len(newframes),len(frames)))
			frames = newframes

		framenames.extend([os.path.splitext(os.path.basename(l))[0] for l in frames])

		# all frame for this video, get all anno
		for frame in frames:
			anno = dict(np.load(frame))

			labels = []
			boxes = []
			for i,classname in enumerate(list(anno['labels'])):
				if targetClass2id.has_key(classname):
					labels.append(targetClass2id[classname])
					boxes.append(anno['boxes'][i])
				else:
					ignored_classes[classname] = 1
			anno['boxes'] = np.array(boxes,dtype="float32")
			anno['labels'] = labels

			# get stat for each class
			for classname in label_dist:
				num_box_this_img = len([l for l in anno['labels'] if l == targetClass2id[classname]])
				label_dist[classname].append(num_box_this_img)
			label_dist_all.append(len(anno['labels']))

	print ignored_classes
	for classname in label_dist:
		d = label_dist[classname]
		ratios = [a/float(b) for a,b in zip(d, label_dist_all)]
		print "%s, [%s - %s], median %s per img, ratio:[%.3f - %.3f], median %.3f, no label %s/%s [%.3f]"%(classname, min(d), max(d), np.median(d), min(ratios), max(ratios), np.median(ratios), len([i for i in d if i==0]), len(d),len([i for i in d if i==0])/float(len(d)))
	print "each img has boxes: [%s - %s], median %s"%(min(label_dist_all),max(label_dist_all),np.median(label_dist_all),)

	# get all the frame that has at least one label for each class in select
	selects_frames = {}
	for classname in selects:
		selects_frames[classname] = []
		assert len(framenames) == len(label_dist[classname])
		for framename,count in zip(framenames, label_dist[classname]):
			if count >0:
				selects_frames[classname].append(framename)
		total = len(selects_frames[classname])
		random.shuffle(selects_frames[classname])
		selects_frames[classname] = selects_frames[classname][:int(total*selects[classname])]
		print "%s, total non empty frame %s, random got %s, "%(classname, total, len(selects_frames[classname]))

	# union all the selected class frames
	addition_frames = list(set([f for c in selects_frames for f in selects_frames[c]]))

	print "got %s frames from the selected class"%(len(addition_frames))

	skip = args.skip
	order_frames = framenames[::skip]

	final = list(set(order_frames + addition_frames))

	print "total frame %s, skip %s and get %s, add selected and final get %s"%(len(framenames), skip, len(order_frames), len(final))

	with open(args.framelst,"w") as f:
		for one in final:
			f.writelines("%s\n"%one)