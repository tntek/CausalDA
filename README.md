
# CausalDA
<!-- 
Code (pytorch) for ['Unified Source-Free Domain Adaptation']() on Digits(MNIST, USPS, SVHN), Office-Home, VisDA-C, domainnet126.  
-->

### Preliminary

- **Datasets**
  - `office-home` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)
  - `VISDA-C` [VISDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  
You need to download the dataset,  modify the path of images in each '.txt' under the folder './data/'.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.3
- pytorch ==1.6.0
- torchvision == 0.7.0


### Training and evaluation

We provide config files for experiments. 

### Source

- We provide the pre-trained source models which can be downloaded from [here](https://drive.google.com/drive/folders/1nKCKd_hASHbetZBCqWVL2c3ZGyQSL-9p?usp=drive_link).

### Target
After obtaining the source models, modify your source model directory. 
You can also  refer to the file on [run.sh](./test_32.sh).




### Contact

- baiyunxiang11@gmail.com
