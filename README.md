# Automated void detection in high resolution x-ray printed circuit boards (PCBs) images with deep segmentation neural network

This is a PyTorch implementation of [Automated void detection in high resolution x-ray printed circuit boards (PCBs) images with deep segmentation neural network](https://www.sciencedirect.com/science/article/abs/pii/S0952197624005839) by Ho Yeung Ma, Minglu Xia, Ziyang Gao, and Wenjing Ye.

## Used libary
* pytorch 1.12.1
* torchvision 0.13.1
* segmentation-models-pytorch 0.3.0
* cv2 4.5.5
* PIL 9.2.0
* prettytable 3.3.0
## Training
Train all encoder and decoder combinations on dataset 1. 
```
python main.py --dataset_root ./datasets --dataset d1
```
Train the model in a specific set of encoder and decoder, e.g. Resnet34 and UnetPlusPlus.
```
python main.py --dataset_root ./datasets --dataset d1 --encoder Resnet34 --decoder UnetPlusPlus
```
## Eval
Evaluate the trained model, 
```
python eval.py --log_path ./Run/...... --test_path PATH_TO_TEST_DATA --save_path ./Results --dataset_root ./datasets
```
## Transfer Learning
WIP

## Results
![table of results](https://github.com/malexhy/X-ray_PCB_void_detection/blob/92debaaaf85fe8ff87f0e95351c7278ceedd5ffb/Images/results.png)
## Citation
```
@article{MA2024108425,
        title = {Automated void detection in high resolution x-ray printed circuit boards (PCBs) images with deep segmentation neural network},
        journal = {Engineering Applications of Artificial Intelligence},
        volume = {133},
        pages = {108425},
        year = {2024},
        issn = {0952-1976},
        doi = {https://doi.org/10.1016/j.engappai.2024.108425},
        url = {https://www.sciencedirect.com/science/article/pii/S0952197624005839},
        author = {Ho Yeung Ma and Minglu Xia and Ziyang Gao and Wenjing Ye},
}
```
