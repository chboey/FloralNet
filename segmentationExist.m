%-------------------------------------------------------------------
% Loading the files, removing flower images that do not have
% annotations

% Grab the Folder Path of Flower Images Only.
images_folder = 'images_256';
image_path = dir(fullfile(images_folder,'*.jpg'));
% View properties to find which one contains the list of flower images
disp(image_path)
image_filename= {image_path.name};

% Strip the extension of the image file.
strip_extension_image_file = strrep(image_filename,'.jpg','');

% Grab the Folder Path of Label Images Only.
label_folder = 'labels_256';
label_image_path = dir(fullfile(label_folder,'*.png'));
% View properties to find which one contains the list of label images
disp(image_path)
label_filename = {label_image_path.name};

% Strip the extension of the label file.
strip_extension_label_file = strrep(label_filename,'.png','');

% To mitigate the problem of file imbalance, gather the list of files
%that are present in images_256 that do not have annotation.
difference = setdiff(strip_extension_image_file,strip_extension_label_file);
disp(difference)

% Delete files in images_256 folder by looping through the list of difference (files that exist
% in images_256 that is not present in labels_256)

for i = 1:numel(difference)
    diff_file = fullfile(images_folder,[difference{i},'.jpg']);
    if exist(diff_file,'file')
        delete(diff_file);
    end
end

% To confirm that: for each image that exist in images_256, their respective
% label is present. Nothing would be displayed as it would be balanced
% already. At this point there is an equal number of files in labels_256 and
% images_256
disp(difference)

%-------------------------------------------------------------------
% Prepare imageDatastore and pixelLabelDatastore for network training
% Author: MathWorks
% Source: “Computer Vision Toolbox Documentation - MathWorks United Kingdom.” 
%          https://uk.mathworks.com/help/vision/ref/pixellabeldatastore.html.


% Create image datastore
imds = imageDatastore(images_folder);

% Define class names and corresponding pixel label IDs
% The boundaries interested are: Flower and Background (Pixel ID: 1 and 3)
% as mentioned in the coursework
classnames = {'Flower', 'Background'};
pixelLabelID = [1, 3];

% Create pixel label datastore
pxds = pixelLabelDatastore(label_folder, classnames, pixelLabelID);

% Splitting and Combining Datastores for network training
% Author : MathWorks
% Source : MathWorks, “Semantic segmentation using Deep Learning - MATLAB & Simulink - MathWorks United Kingdom.” 
%          https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html

% This section of the code was derived from the function
% "partitionCamVidData(imds,pxds)" found in the citated website, but the
% ratio chosen was different: 70% for training data 20% for validation and
% 10% for test data

% Define the ratios for splitting
train_ratio = 0.7;
val_ratio = 0.2;
test_ratio = 0.1;

% Count the total number of images
num_images = numel(imds.Files);

% Shuffle indices for stochasticity 
rng(1); 
shuffled_indices = randperm(num_images);

% Calculate the number of images for each set
num_train = floor(train_ratio * num_images);
num_val = floor(val_ratio * num_images);
num_test = num_images - num_train - num_val;

% Split the indices
train_indices = shuffled_indices(1:num_train);
val_indices = shuffled_indices(num_train+1:num_train+num_val);
test_indices = shuffled_indices(num_train+num_val+1:end);

% Create image datastore splits
imdsTrain = subset(imds, train_indices);
imdsVal = subset(imds, val_indices);
imdsTest = subset(imds, test_indices);

% Create pixel datastore splits
pxdsTrain = subset(pxds, train_indices);
pxdsVal = subset(pxds, val_indices);
pxdsTest = subset(pxds, test_indices);

% After splitting, inbuilt function combine can be used to combine the
% respective split sets of imds and pxds
dsTrain = combine(imdsTrain,pxdsTrain);
dsVal = combine(imdsVal,pxdsVal);
dsTest = combine(imdsTest,pxdsTest);

%-------------------------------------------------------------------
% Initialising DeepLabv3+ Network with initialised weights from ResNet 18
% and ResNet 50 (Per Iteration)
% Author : MathWorks
% Source : MathWorks, “Semantic segmentation using Deep Learning - MATLAB & Simulink - MathWorks United Kingdom.” 
%          https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
% Create Network 
imagesize = [256 256 3];
numClasses = numel(classnames);
% First Iteration would be ResNet 18, Second Iteration would be ResNet50
network = "resnet18"; % replace resnet18 with resnet50

% Initialise a DeepLabv3+ network based on the initialised backbone network
% (ResNet 18 / ResNet 50)
% Input arguments
% imagesize = 256, 256, 3 (RGB Image with dimensions of 256 by 256)
% Configures input layer to match image size
%
%
% numClasses = 2
% Number of classes for the network to classify, configures output layer to
% give 2 outputs.
net = deeplabv3plus(imagesize,numClasses,network);

% Training options were derived from the citated source in this section,
% with tweaks such as changing the max epochs to 10 

% Define training options
options = trainingOptions("sgdm",...
    LearnRateSchedule="piecewise",...
    LearnRateDropPeriod=6,...
    LearnRateDropFactor=0.1,...
    Momentum=0.9,...
    InitialLearnRate=1e-2,...
    L2Regularization=0.005,...
    ValidationData=dsVal,...
    MaxEpochs=10,...  
    MiniBatchSize=4,...
    Shuffle="every-epoch",...
    CheckpointPath=tempdir,...
    VerboseFrequency=10,...
    ValidationPatience=4);


% Train network, with normalised cross entropy that handles null values
% The function lies in modelLoss.m file, which was derived from the source
% citated in this section
net = trainnet(dsTrain,net,@modelLoss,options);

% Export the network file, when loaded into the workspace it would be
% initialised as net
save('segmentexistnet.mat', 'net');

%-------------------------------------------------------------------
% Evaluate Segmentation Networks
% Author: MathWorks
% Source: “Evaluate and inspect results of semantic segmentation - MATLAB & Simulink - MathWorks United Kingdom.” 
%           https://uk.mathworks.com/help/vision/ug/evaluate-and-inspect-the-results-of-semantic-segmentation.html

% Load all images from the test set and perform semantic segmentation.
pxdsTestResults = semanticseg(imdsTest,net,Classes=classnames,WriteLocation=tempdir);

% Evaluate based on predicted pixels and truth labels
metrics = evaluateSemanticSegmentation(pxdsTestResults,pxdsTest);

% Display Class Metrics of Accuracy, IoU, MeanBFScore, and the Confusion
% Matrix
disp(metrics.ClassMetrics)

disp(metrics.ConfusionMatrix)

% Display Confusion Matrix with the percentage of TP, FP, TN, FN
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classnames,Normalization="row-normalized");
cm.Title = "Normalized Confusion Matrix (%)";

% Calculate Mean IoU
imageIoU = metrics.ImageMetrics.MeanIoU;
disp(imageIoU)


% Find Worst/Best Predicted Image 
[minIoU, worstImageIndex] =max(imageIoU); % Change this to min to find the worst
minIoU = minIoU(1);
worstImageIndex = worstImageIndex(1); % Find index of worst predicted image based on low mean 
% IoU score

% For that particular image, find the actual image before segmentation, the
% truth labels and the predicted labels.
worstTestImage = readimage(imdsTest,worstImageIndex);
worstTrueLabels = readimage(pxdsTest,worstImageIndex);
worstPredictedLabels = readimage(pxdsResults,worstImageIndex);

% Truth labels from the image are derived by converting binary array into
% greyscale image, where truth values are represented by white pixels.
worstTrueLabelImage = im2uint8(worstTrueLabels == classnames(1));
worstPredictedLabelImage = im2uint8(worstPredictedLabels == classnames(1));

% Display the images side by side using subplot ( In the citated source,
% they utilised montages but this research used subplots for its easy
% usage )

% First Image from the Left
subplot(1, 3, 1); 
imshow(worstTestImage); 
title('Test Image'); 

% Middle Image
subplot(1, 3, 2); 
imshow(worstTrueLabelImage); 
title('True Label Image'); 

% Last Image
subplot(1, 3, 3); % Create the third subplot
imshow(worstPredictedLabelImage); % Display the third image
title('Predicted Label Image');







