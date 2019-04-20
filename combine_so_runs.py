# coding=utf-8
# given the original json output and the small object output path, combine into one outpath

import sys, os, argparse, json
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("oripath")
parser.add_argument("sopath")
parser.add_argument("outpath")

if __name__ == "__main__":
	args = parser.parse_args()

	if not os.path.exists(args.outpath):
		os.makedirs(args.outpath)

	orifiles = glob(os.path.join(args.oripath, "*.json"))

	so_objects = ["Prop", "Push_Pulled_Object", "Prop_plus_Push_Pulled_Object", "Bike"]
	for orifile in tqdm(orifiles, ascii=True):
		filename = os.path.splitext(os.path.basename(orifile))[0]
		sofile = os.path.join(args.sopath, "%s.json"%filename)

		with open(orifile, "r") as f:
			data = json.load(f)

		# 1. get all the non so_objects from original outputs
		newdata = []
		for one in data:
			if one['cat_name'] not in so_objects:
				newdata.append(one)

		with open(sofile, "r") as f:
			data = json.load(f)

		for one in data:
			if one['cat_name'] in so_objects:
				newdata.append(one)

		newfile = os.path.join(args.outpath, "%s.json"%filename)
		with open(newfile, "w") as f:
			json.dump(newdata, f)



