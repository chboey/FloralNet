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
<img src="https://github.com/chboey/FloralNet/assets/103494565/f9a4ebde-a172-4fbb-8e38-6edd76b033a1" alt="ResNet18" width="700" height="150">

### Confusion Matrices
## Confusion Matrix for DeepLabv3+ Initialized with ResNet 18 weights

| True\Predicted | Background | Flower |
|----------------|------------|--------|
| Background     | 99.6%      | 0.4%   |
| Flower         | 1.6%       | 98.4%  |

## Confusion Matrix for DeepLabv3+ Initialized with ResNet 50 weights

| True\Predicted | Background | Flower |
|----------------|------------|--------|
| Background     | 99.6%      | 0.4%   |
| Flower         | 1.4%       | 98.6%  |

## Confusion Matrix for FloralNet

| True\Predicted | Background | Flower |
|----------------|------------|--------|
| Background     | 97.8%      | 2.2%   |
| Flower         | 3.5%       | 96.5%  |
 <img src="https://github.com/chboey/FloralNet/assets/103494565/5119eb2b-12f6-412c-bb82-1a82ebefca46" alt="ResNet18" width="270" height="300">

## Visualizations
<table>
  <tr>
    <td>
      <img src="https://github.com/chboey/FloralNet/assets/103494565/d876a24b-069e-4eec-930e-59ac446c4af3" alt="ResNet18" width="250" height="200">
    </td>
    <td>
      <img src="https://github.com/chboey/FloralNet/assets/103494565/924d634c-51bd-48e9-a846-0769df94dd69" alt="ResNet50" width="250" height="200">
    </td>
    <td>
      <img src="https://github.com/chboey/FloralNet/assets/103494565/caf88d1d-c783-4669-a1fe-5edf6ce5836f" alt="Floral-Net" width="250" height="200">
    </td>
  </tr>
</table>




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
