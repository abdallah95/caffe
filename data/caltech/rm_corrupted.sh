#!/bin/sh

# this bash code will remove all image files that are corrupted (their file size is 0). 

for dataset in trainval test
do
	for imagefile in $dataset/images/*.jpg
	do
		image_size=$(wc -c < $imagefile);
		if [ "$image_size" -eq "0" ]
		then
			rm $imagefile
			txtfile=${imagefile##*/}
			txtfile=$dataset/"annotations/"${txtfile%.jpg}".txt"
			rm $txtfile
		fi
	done
done