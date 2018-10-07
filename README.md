# CSI-Net pytorch
This code can achieve multi-channel time-serial signals classification  as well as regression by modifying modern backbone networks. In our original paper, we have achieved four human sensing tasks with WiFi signals.
 
We have applied this code with little modification to do motor fault diagnosis in engineering industry. Paper has been submitted to the TRANSACTIONS ON INDUSTRIAL ELECTRONICS with a title *Multi-scale Residual Learning Convolutional Neural Network for End-to-end Motor Fault Diagnosis Under Nonstationary Conditions*.

Currently, we are designing a light-weighted version for sensing tasks with COTS RFID. We believe this generic deep networks can solve more sensory data tasks. 

## Tested Environment
1. python 3.6
1. pytorch 4.0.1
2. cuda 8.0/9.0
3. Windows7/Ubuntu 16.04

## Useage
1. clone repository: *git clone https://github.com/geekfeiw/CSI-Net.git*
2. *mkdir weights*, then download pre-trained model into *weights*: *https://drive.google.com/open?id=16HOqFagtigjGry5asx-ErradnWS5rfe2*

We used model, *model/res_net_use_this.py*, to jointly solve two body characterization tasks. Besides, we provide *model/solo_task_res_net.py* for solo classification tasks. 


## Citation
Please cite this paper in your publications if it helps your research. On hold in
@article{wang2018csi,
  title={CSI-Net: Unified Body Characteration and Action Recognition},
  author={Wang, Fei and Han, Jinsong and Zhang, Shiyuan and He, Xu, Huang, Dong},
  journal={arXiv preprint},
  year={2018}
}

