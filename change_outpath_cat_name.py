# coding=utf-8
# temporay fix, change all the json cat_name according to cat id
# make the mistake of conveting the class to cooco class
import sys, os, argparse, json
from glob import glob
from tqdm import tqdm
from class_ids import targetClass2id_new_nopo as targetClass2id
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument("outpath")
parser.add_argument("cocojson")
parser.add_argument("newpath")

if __name__ == "__main__":
	args = parser.parse_args()

	outfiles = glob(os.path.join(args.outpath, "*.json"))

	if not os.path.exists(args.newpath):
		os.makedirs(args.newpath)

	coco = COCO(args.cocojson)
	cat_ids = coco.getCatIds() #[80], each is 1-90
	classId_to_cocoId = {(i+1):v for i,v in enumerate(cat_ids)} # we used this in the forward
	cocoId_to_classId = {v:(i+1) for i,v in enumerate(cat_ids)}

	targetid2class = {v:k for k,v in targetClass2id.iteritems()}

	for outfile in tqdm(outfiles, ascii=True):
		with open(outfile, "r") as f:
			data = json.load(f)

		newdata = []
		for one in data:
			one['cat_name'] = targetid2class[cocoId_to_classId[one['category_id']]]

			newdata.append(one)

		filename = os.path.splitext(os.path.basename(outfile))[0]
		target_file = os.path.join(args.newpath, filename+".json")
		with open(target_file, "w") as f:
			json.dump(newdata, f)