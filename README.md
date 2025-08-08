# Automated void detection in high resolution x-ray printed circuit boards (PCBs) images with deep segmentation neural network

This is a PyTorch implementation of [Automated void detection in high resolution x-ray printed circuit boards (PCBs) images with deep segmentation neural network](https://www.sciencedirect.com/science/article/abs/pii/S0952197624005839) by Ho Yeung Ma, Minglu Xia, Ziyang Gao, and Wenjing Ye.

## Dataset
Please contact hymaaf@connect.ust.hk for the dataset used in the paper.

## Training
Train all encoder and decoder combinations on dataset 1. 
```
python main.py --dataset_root ./datasets --dataset d1
```
Train the model in a specific set of encoder and decoder, e.g. Resnet34 and UnetPlusPlus.
```
python main.py --dataset_root ./datasets --dataset d1 --encoder Resnet34 --decoder UnetPlusPlus
```
## Transfer Learning
WIP

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
