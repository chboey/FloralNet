# 🌸 Comparative Analysis of Segmentation Performance

## Overview

This project delves into the applications of deep learning Convolutional Neural Networks (CNNs): DeepLab v3+ and a CNN inspired by U-Net called Floral-Net, 
tasked with segmenting flowers and backgrounds from the Oxford Flower Dataset.

## Methodology

This study addresses imbalanced data in the Oxford Flower Dataset by carefully preprocessing images and labels. Using MATLAB, missing labels are removed, resulting in 846 image-label pairs. Transfer learning with DeepLabv3+ using ResNet18 and ResNet50 is employed, alongside a unique Floral-Net architecture inspired by prior works. Floral-Net integrates incremental dropout, selective filter adjustments, and weighted pixel classification to mitigate class imbalance. This comprehensive approach aims to improve semantic segmentation of flower images.

## Dataset
The Oxford Flower Dataset (17 Classes) was used.

## Training Options

### DeepLabv3+ Initialised with ResNet18/50 weights

| Option                    | Value          |
|---------------------------|----------------|
| Optimization Algorithm   | sgdm           |
| Learning Rate Schedule   | piecewise      |
| Learning Rate Drop Period| 6              |
| Learning Rate Drop Factor| 0.1            |
| Momentum                 | 0.9            |
| Initial Learning Rate    | 0.01           |
| L2 Regularization        | 0.005          |
| Validation Data          | dsVal          |
| Max Epochs               | 10             |
| Mini-Batch Size          | 4              |
| Shuffle                  | every-epoch    |
| Checkpoint Path          | tempdir        |
| Verbose Frequency        | 10             |
| Validation Patience      | 4              |

### Floral-Net

| Option                    | Value          |
|---------------------------|----------------|
| Optimization Algorithm   | Adam           |
| Initial Learning Rate    | 0.001          |
| Learning Rate Schedule   | Piecewise      |
| Learning Rate Drop Factor| 0.1            |
| Learning Rate Drop Period| 10             |
| L2 Regularization        | 0.0001         |
| Maximum Epochs           | 10             |
| Mini-Batch Size          | 16             |
| Data Shuffling           | Every Epoch    |
| Validation Data          | dsVal          |
| Validation Frequency     | 10             |

## Networks
- DeepLab v3+: Initialized with ResNet18 and ResNet50.
- Floral-Net: A simplified Encoder-Decoder Network with layers mirroring U-Net

Link to Networks : [Link](https://drive.google.com/file/d/1cxY6ojJGHyVY0A3SjUIdA94DgrClM0qa/view?usp=sharing)

## Results

### Performance Metrics
<img src="https://github.com/chboey/FloralNet/assets/103494565/f9a4ebde-a172-4fbb-8e38-6edd76b033a1" alt="ResNet18" width="700" height="150">

### Confusion Matrices
<img src="https://github.com/chboey/FloralNet/assets/103494565/5119eb2b-12f6-412c-bb82-1a82ebefca46" alt="ResNet18" width="370" height="400">

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

