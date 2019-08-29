see-in-the-dark
Learning-to-see-in-the-dark

train model:

1. run_sid_sa.sh: for training the model with the proposed method
2. train_sid_sa.py: train the model
3. models2.py: the architecture of the proposed model
4. load_data.py: use pytorch to read the dataset
  -  prob.txt: 
      1. the text file of the exposure ratio of the training data. The first row is the file name. The second, third, fourth            rows correspond to the number of files with the exposure ratio 0.1, 0.4, and 0.033.
      2. Because we want data with different exposure ratios are uniformly chosen, we use this file to calculate the        probability.
run.sh: for training the learning-to-see-in-the-dark model
train.py: train the model

Test model:

1. Test_sid_sa.py: test the proposed model
2. Test_sid.py: test learning-to-see-in-the-dark model

Requirements:
1. python/3.5.2
2. tensorflow/python3.x/gpu/r1.4.0-py3
3. cuda80/toolkit/8.0.61
4. cuDNN/cuda80/6.0.21
5. scipy
6. numpy
7. rawpy
8. Scikit-Image

Dataset:
Sony subdataset in SID dataset(https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx)

Citations:
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.


