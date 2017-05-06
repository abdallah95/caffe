import sys, os, argparse, time
import pdb

import numpy as np
import scipy.io

# this script was used to extract features for quantization


def get_args():
	parser = argparse.ArgumentParser('Extract CNN weights')

	parser.add_argument('-m', dest='modelFile', type=str, required=True,
						help='Caffe model weights file to parse')
	parser.add_argument('-n', dest='netFile', type=str, required=True,
						help='Network prototxt file associated with model')
	parser.add_argument('-i', dest='iterations', type=int, required=True,
						help='Number of iterations')

	return parser.parse_args()




if __name__ == "__main__":
	import caffe

	args = get_args()
	net = caffe.Net(args.netFile, args.modelFile, caffe.TEST)
	iterations = args.iterations

	file_output = open('ssd_features.txt','w+')

	caffe.set_mode_gpu()

	max_f = {}
	min_f = {}

	for iteration in range(iterations):
		net.forward()
		for name, blobs in net.params.iteritems():
			print("Iteration #"+str(iteration)+": Extracting " + name + " features")
			features = net.blobs[name]

			if not name in max_f:
				max_f[name] = -(1<<32)
				min_f[name] = (1<<32)

			tmp_max = max_f[name]
			tmp_min = min_f[name]
			for i in features.data:
				for j in i:
					for k in j:
						for f in k:
							max_f[name] = max(max_f[name],abs(f))
							min_f[name] = min(min_f[name],abs(f))

			if max_f[name] != tmp_max:
				print("Max value updated. Difference: "+str(max_f[name]-tmp_max))
			if min_f[name] != tmp_min:
				print("Min value updated. Difference: "+str(min_f[name]-tmp_min))


	for name in max_f:
		file_output.write(name + ': min = '+str(min_f[name])+' max = '+str(max_f[name]) + '\n')

print("Features extracted successfully")

file_output.close()
