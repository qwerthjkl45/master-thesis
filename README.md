# master-thesis

## Goal: 
  - Use an end-to-end learned based method to do the image enhancement task: **image quality enhancement, extreme low-light image enhancement task**
  - Meanwhile, solve the problem of the **artifacts** in the images produced by the state-of-the-art methods (DPED and SID).
  
## Method: 
 - Use U-Net GAN as the backbone
 - To further remove artifacts, our proposed model include **two components to extract the global information**(histogram, brightness, scene categories, etc.) in the images
    - Self attention modules
    - Global feature vector
## Result:
  - In the task of image quality enhancement:
    1. In term of PSNR, the proposed model performs better compared to the baseline DPED
    2. Enhanced images in the proposed model have less artifacts and noise
    3. Example image in the DPED dataset:
  - In the task of extreme low-light image enhancement:
    1. In qualitative evaluation, the proposed model generated images with more texture details and less artifacts
    2. Example image in the SID dataset:
      
## Defense Slides:  
  https://docs.google.com/presentation/d/1lXiYRm-Tf6IlyN0lCGF2MXJr4gIzHSIpvdx-YEA392c/edit?usp=sharing

## pretrained models 
  (https://drive.google.com/open?id=1Bj9PABfD5eftHeM4ifE4s8UAmvpVn6Qj): 
  - move model/dped/* to ./DPED/models/
  - move model/sid/* to ./sid/sid_w_sa/
  
