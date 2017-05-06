import sys, os, argparse, time
import pdb

import numpy as np
import scipy.io

# this script was used to extract weights for quantization

def get_args():
	parser = argparse.ArgumentParser('Extract CNN weights')

	parser.add_argument('-m', dest='modelFile', type=str, required=True,
						help='Caffe model weights file to parse')
	parser.add_argument('-n', dest='netFile', type=str, required=True,
						help='Network prototxt file associated with model')

	return parser.parse_args()




if __name__ == "__main__":
	import caffe

	args = get_args()
	net = caffe.Net(args.netFile, caffe.TRAIN, weights=args.modelFile)

	file_output = open('ssd_weights.txt','w+')
	for name, blobs in net.params.iteritems():
		for ii in range(len(blobs)):
			# Assume here index 0 are the weights and 1 is the bias.
			# This seems to be the case in Caffe.
			if ii == 0:
				name2 = name + "Weight"
			elif ii == 1:
				name2 = name + "Bias"
			else:
				pass # This is not expected
				
			max_w = -(1<<32)
			min_w = 1<<32

			print("%s" % (name2))
			shape = len(blobs[ii].data.shape)
			if shape == 4:
				for i in blobs[ii].data:
					for j in i:
						for k in j:
							for w in k:
								max_w = max(max_w,abs(w))
								min_w = min(min_w,abs(w))
			else:
				for w in blobs[ii].data:
					max_w = max(max_w,abs(w))
					min_w = min(min_w,abs(w))

			file_output.write(name2 + ': min = '+str(min_w)+' max = '+str(max_w) + '\n')
			# file_output.write(name2 + ': min = '+"{:.40f}".format(min_w)+' max = '+"{:.40f}".format(max_w) + '\n')
 #          scipy.io.savemat(name2+".mat", {name2 : blobs[ii].data})

file_output.close()
