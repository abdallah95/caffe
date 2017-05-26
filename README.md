# SSD: Single Shot MultiBox Detector On Caltech Pedestrian Dataset

### Introduction

In this work we apply Single Shot Multibox Detector [SSD](https://github.com/weiliu89/caffe/tree/ssd/) on Caltech Pedestrian Dataset. In addition to caltech dataset, we also used ETH pedestrian dataset and TUDBrussels dataset. We are also finetuning from SSD512 model trained on 07++12+COCO. We were able to reach state-of-art results while having a real-time speed.

Results are shown below:

| Model | Overall miss-rate | Reasonable miss-rate | **FPS** (Geforce GTX Titan X) | Input resolution
|:-------|:-----:|:-------:|:-------:|:-------:|
| SSD512 (VGG16) (training from scratch + no hyper-parameters optimization) | 65.17% | 20.32%  | 22 | 640 x 480 |
| SSD512 (VGG16) | 54.44% | 11.89% | 24 | 512 x 512 |
| SSD640 (VGG16) | 53.11% | 11.85%  | 20 | 640 x 480 |
| F-DNN | 50.5% | 8.65%  | 6.25 | 640 x 480 |

### Fixed-Point 16-bit Quantization
We also worked on quantizing the model to dynamic 16-bit Fixed Point using [caffe ristretto](http://lepsucd.com/?page_id=621). The script to test the quantized model is available under `ssd-ristretto` branch by going to `models/VGGNet/caltech/SSD_512x512_ft_quantized` (Caffe ristretto doesn't require changing the .caffemodel file for quantization, only the .prototxt file is modified). The model performance decreased by less than 0.01%. We do not report the speed on the Quantized model, Ristretto simulates the 16-bit fixed point arithmetic using floating point arithmetic because there's no hardware support for fixed point arithmetic on the GPU, but with hardware support we expect the model to be faster.

| Model | Overall miss-rate | Reasonable miss-rate
|:-------|:-----:|:-------:
| SSD512 (VGG16) not quantized | 54.4362% | 11.8868%
| SSD512 (VGG16) quantized | 54.4374% | 11.8937%

### Citing

Please cite this paper in your publications if it helps your research:
    
    @inproceedings{feasac2017ssdc,
      title = {An FPGA-Accelerated Design for Deep Learning Pedestrian Detection in Self-Driving Vehicles},
      author = {Moussawi, Abdallah and Haddad, Kamal and Chahine, Anthony},
      booktitle = {FEASAC},
      year = {2017}
    }

and of course, please cite the great work done by Wei Liu et. al:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }
    
    
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/amoussawi/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

### Preparation
examples/ssd/ contains python scripts to train with two initializations: with finetuning (ends with \_ft) and without finetuning.

1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6) If you want to start training from scratch, and [SSD512 07++12+COCO](https://drive.google.com/open?id=0BzKzrI_SkD1_NVVNdWdYNEh1WTA) if you want to finetune. Atrous VGGNet should be stored in `$CAFFE_ROOT/models/VGGNet/`. And pretrained SSD512 model should be stored in `$CAFFE_ROOT/models/VGGNet/VOC0712Plus/SSD_512x512/`

2. Download Caltech, ETH, and TUDBrussels pedestrian datasets from [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). By default, we assume the data is stored in `$HOME/data/caltech_code/`

4. You need Matlab in order to use the caltech evaluation code. The code is available in data/caltech/caltech_code/. We extract 1 frame every 5 frames from caltech training dataset, and all frames of ETH and TUDBrussels datasets. (we also used the external ETH car dataset [here](https://data.vision.ee.ethz.ch/cvl/aess/dataset/) consisting of ~2100 pedestrians, though we don't think it made a huge difference, so you may not need it). To extract datasets, you need to run `extractDatasets.m` matlab script in `data/caltech/caltech_code`. This will extract the datasets into `../trainval/` and `../test/` accordingly. If you want to extract more images from caltech dataset, just set the `skip` variable of `usatrain` in `dbInfo.m` accordingly.

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/caltech/
  ./data/caltech/create_list.sh
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $CAFFE_ROOT/data/caltech/caltech_trainval_lmdb
  #   - $CAFFE_ROOT/data/caltech/caltech_test_lmdb
  # and make soft links at examples/caltech/
  ./data/caltech/create_data.sh
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/caltech/SSD_512x512/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/caltech/SSD_512x512/
  # and save temporary evaluation results in:
  #   - $CAFFE_ROOT/examples/results/SSD_512x512/
  # It should reach 11.8* % at 20k iterations.
  python examples/ssd/ssd_caltech_512_ft.py
  ```

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_caltech_ft.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_caltech_webcam_ft.py
  ```
  [Here](https://youtu.be/iKIW5Q0XAcg) is a demo video of running a SSD512 model on a video of a car driving in the streets of Beirut.

### Models
[SSD512 Caltech](https://www.dropbox.com/s/1zvmbj2gtchdxhm/SSD_512x512_ft.zip?dl=0)
