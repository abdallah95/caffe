# cd /home/abdallah/FYP/caffe-ristretto-ssd
./build/tools/caffe train \
--solver="models/VGGNet/caltech/SSD_512x512_ft_quantized/solver.prototxt" \
--weights="models/VGGNet/caltech/SSD_512x512_ft_quantized/VGG_caltech_SSD_512x512_ft_iter_20000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/caltech/SSD_512x512_ft_score/VGG_caltech_SSD_512x512_ft_test20000.log
