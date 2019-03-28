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
