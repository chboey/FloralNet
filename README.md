modelLoss.m File
-----------------
Custom model loss function: Normalised cross entropy that handles null values
Author : MathWorks
Source : MathWorks, “Semantic segmentation using Deep Learning - MATLAB & Simulink - MathWorks United Kingdom.” 
          https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html



Caution when Loading the Network Files
--------------------------------------

When loading either network file, they would be both initialized as net (as per coursework requirement). Therefore, it would be recommended to clear the workspace entry for net when you are about to test another network.

Thank you!


data.zip
---------------------

Contains the dataset which now contains 846 flower images, and 846 label images. This ensures that each flower image has a label image. Preprocessing steps for this is located at the beginning of "segmentationExist.m" and "segmentationOwn.m"