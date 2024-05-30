# 🌸 Comparative Analysis of Segmentation Performance

## Overview
This project delves into the applications of deep learning Convolutional Neural Networks (CNNs): DeepLab v3+ and a CNN inspired by U-Net called Floral-Net, 
tasked with segmenting flowers and backgrounds from the Oxford Flower Dataset.

# Methodology

## Dataset
The Oxford Flower Dataset (17 Classes) was used.

## Training Options
| Option          | Value  |
|-----------------|--------|
| InitialLearnRate| 0.001  |
| MaxEpochs       | 50     |
| MiniBatchSize   | 4      |

## Networks
- DeepLab v3+: Initialized with ResNet18 and ResNet50.
- Floral-Net: A modified U-Net architecture.

## Results

### Performance Metrics

| Network                | Accuracy | IoU   | BF Score |
|------------------------|----------|-------|----------|
| DeepLab v3+ (ResNet18)| 98.35%   | 95.12%| 93.50%   |
| DeepLab v3+ (ResNet50)| 99.61%   | 97.45%| 95.72%   |
| Floral-Net             | 96.50%   | 92.35%| 81.15%   |

### Confusion Matrices

| Network   | Flower Misclassified | Background Misclassified |
|-----------|----------------------|--------------------------|
| ResNet18  | 1.8%                 | 0.6%                     |
| ResNet50  | 1.4%                 | 0.4%                     |
| Floral-Net| 2.2%                 | 3.5%                     |

## Visualizations
![image](https://github.com/chboey/FloralNet/assets/103494565/d876a24b-069e-4eec-930e-59ac446c4af3)


## Conclusion
- DeepLab v3+ with ResNet50 outperformed other networks.
- Floral-Net is a strong contender with its lightweight architecture.

## Future Work
- Experiment with more augmentations.
- Explore other lightweight architectures.

## References
- S. Saha, et al., "Self-supervised Deep Learning for Flower Image Segmentation," 2020.
- Y. Wu, et al., "Convolution Neural Network based Transfer Learning for Classification of Flowers," 2018.
- O. Ronneberger, et al., “U-NET: Convolutional Networks for Biomedical Image Segmentation,” 2015.
