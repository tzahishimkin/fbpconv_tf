# FBPConvNet - tensorflow
http://ieeexplore.ieee.org/document/7949028/

This is tensorflow implementation for ``Deep Convolutional Neural Network for Inverse Problems in Imaging, TIP (2017)``.
- applications (forward model with shift-invariant normal operator):
* 2D sparse-view CT reconstruction 
* reconstruction of accelerated MRI
* Deconvolution of shift-invariant

Whole codes are forked and modified from https://github.com/jakeret/tf_unet.

## Training configuration
* Tensorflow 1.1.0
* 1 or 2 GPUs (TITAN X pascal arch.)
* MacOS X 10.12.6
* Python 2.7.12

## Data - XYCN format (ellipsoids, downsampling factor : x 20)
* train : https://drive.google.com/open?id=1FTOgM2vOQaGSokEDtOaPNdBTto6h5yFi
* test : https://drive.google.com/open?id=1w_kPao6L2UwhTKIgcr_3o62A6vYYtX_r
* If you want to make fbp images, you can find file_generator in tf_unet/layers.py (load_whole_data function.)

### illustration
![alt tag](https://github.com/panakino/fbpconv_tf/blob/master/structure.png)

## Commands
Before starting,
```bash
pip install pillow matplotlib scipy scikit-image h5py
```

To start training a model for FBPConvNet:
```bash
python main.py --lr=1e-4 --output_path='logs/' --train_path='train_data/*.mat' --test_path='test_data/*.mat' --features_root=32 --layers=5 
```

To deploy trained model:
```bash
python main.py --lr=1e-4 --output_path='logs/' --train_path='train_data/*.mat' --test_path='test_data/*.mat' --features_root=32 --layers=5 --is_training=False
```

You may find more details in main.py.

## Other papers, code and overleaf project

Overleaf project
https://www.overleaf.com/17262924fkvbtvxhkxgq#/65727478/

The paper for this inplementation is at: https://arxiv.org/abs/1611.03679

Implemention TF solution using architecture with U-Net.
You can find another implementation of this article at:
https://github.com/mughanibu/Deep-Learning-for-Inverse-Problems


A different deep learning solution which seem good:
https://github.com/tzahishimkin/ScaDec-deep-learning-diffractive-tomography


paper:
https://arxiv.org/abs/1803.06594

Also these guys did some analysis on this solution:
https://arxiv.org/pdf/1806.08015.pdf



## Contact
kyonghwan.jin@gmail.com


