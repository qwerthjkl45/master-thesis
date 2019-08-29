need to download imagenet-vgg-verydeep-19.mat to vgg_pretrained/

train model:
1. run run.sh: to start training the model with Unet architecture generator
2. run run_with_sa.sh: to start training the model with Unet architecture generator + self attention mechanism
3. dataset: 
 - training input: dped_dir + 'iphone/*' 
 - training output: dped_dir + 'canon/*'

test model:
1. run test_model.sh to start testing
2. dataset:
 - test input: test_dir/*
 
 
Requirements:
1. python/3.5.2
2. tensorflow/python3.x/gpu/r1.4.0-py3
3. cuda80/toolkit/8.0.61
4. cuDNN/cuda80/6.0.21
5. scipy
6. numpy
7. rawpy
8. Scikit-Image

Citations:
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}

