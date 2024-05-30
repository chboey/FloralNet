%-------------------------------------------------------------------
% Loading the files, removing flower images that do not have
% annotations

% Grab the Folder Path of Flower Images Only.
images_folder = 'images_256';
image_path = dir(fullfile(images_folder,'*.jpg'));
image_filename= {image_path.name};

% Strip the extension of the image file.
strip_extension_image_file = strrep(image_filename,'.jpg','');

% Grab the Folder Path of Label Images Only.
label_folder = 'labels_256';
label_image_path = dir(fullfile(label_folder,'*.png'));
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
%--------------------------------------------------------------------------

% Lightweight Encoder and Decoder Network derived from the layers of U-Net's Encoders
% and Decoders with the following changes:

% 1) Added batch normalisation between each convolution layer and relu
% layer.
%
% 2) Added dropout layer that increments from 0.1 onwards ( encoder ) by 0.1 ,
% at the bottleneck it resets to 0.1, the first decoder that connects from
% the bottleneck would be reinitialsied with dropout probability of 0.4 and
% decrementing from 0.4 by 0.1 onwards. Dropout layers are added to the end
% of each block.
%
% 3) Removed skip connections due to the small size of this segmentation
% application.
%
% 4) Only the last two encoders would have their filters double in numbers, 
%    this trend in consistent in the decoders as well, 
%    where the last two would have their number of filters halve in numbers.
%
% 5) Number of strides were kept consistent at [1,1] (default value so it
% was not initialised in the convolution2dlayers)

% Read the report to see where this network got its inspiration from

numClasses=2;

input_img = [
    imageInputLayer([256 256 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_A')];

encoder1 = [
    convolution2dLayer(3, 32, 'Padding', 'same','Name', 'conv1_1')
    batchNormalizationLayer('Name', 'bnE1_1')
    reluLayer('Name', 'relu1_1')

    convolution2dLayer(3, 32, 'Padding', 'same','Name', 'conv1_2')
    batchNormalizationLayer('Name', 'bnE1_2')
    reluLayer('Name', 'relu1_2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    dropoutLayer(0.1, 'Name', 'drop1') 
    ];

encoder2 = [
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2_1')
    batchNormalizationLayer('Name', 'bnE2_1')
    reluLayer('Name', 'relu2_1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2_2')
    batchNormalizationLayer('Name', 'bnE2_2')
    reluLayer('Name', 'relu2_2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    dropoutLayer(0.2, 'Name', 'drop2')
    ];

encoder3 = [
    convolution2dLayer(3, 64, 'Padding', 'same','Name', 'conv3_1')
    batchNormalizationLayer('Name', 'bnE3_1')
    reluLayer('Name', 'relu3_1')

    convolution2dLayer(3, 64, 'Padding', 'same','Name', 'conv3_2')
    batchNormalizationLayer('Name', 'bnE3_2')
    reluLayer('Name', 'relu3_2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    dropoutLayer(0.3, 'Name', 'drop3')
];

encoder4 = [
    convolution2dLayer(3, 128, 'Padding', 'same','Name', 'conv4_1')
    batchNormalizationLayer('Name', 'bnE4_1')
    reluLayer('Name', 'relu4_1')
    
    convolution2dLayer(3, 128, 'Padding', 'same','Name', 'conv4_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')
    dropoutLayer(0.4, 'Name', 'drop4')
];


bottleneck = [
    convolution2dLayer(3, 128, 'Padding', 'same','Name', 'convB_1')
    batchNormalizationLayer('Name', 'bnBN_1')
    reluLayer('Name', 'reluB_1')

    convolution2dLayer(3, 128, 'Padding', 'same','Name', 'convB_2')
    batchNormalizationLayer('Name', 'bnBN_2')
    reluLayer('Name', 'reluB_2')

    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upB')

    dropoutLayer(0.1, 'Name', 'dropB')
    ];

decoder4 = [
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'convD4_1')
    batchNormalizationLayer('Name', 'bnD4_4')
    reluLayer('Name', 'reluD4_1')

    dropoutLayer(0.4, 'Name', 'dropD4')
];

decoder3 = [
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'convD3_1')
    batchNormalizationLayer('Name', 'bnD3_1')
    reluLayer('Name', 'reluD3_1')

    convolution2dLayer(3, 128, 'Padding', 'same' ,'Name', 'convD3_2')
    batchNormalizationLayer('Name', 'bnD3_2')
    reluLayer('Name', 'reluD3_2')

    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upD3')
    dropoutLayer(0.3, 'Name', 'dropD3')
];


decoder2 = [
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'convD2_1')
    batchNormalizationLayer('Name', 'bnD2_1')
    reluLayer('Name', 'reluD2_1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'convD2_2')
    batchNormalizationLayer('Name', 'bnD2_2')
    reluLayer('Name', 'reluD2_2')

    transposedConv2dLayer(3, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'upD2')
    dropoutLayer(0.2, 'Name', 'dropD2')
];

decoder1 = [
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'convD1_1')
    batchNormalizationLayer('Name', 'bnD1_1')
    reluLayer('Name', 'reluD1_1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'convD1_2')
    batchNormalizationLayer('Name', 'bnD1_2')
    reluLayer('Name', 'reluD1_2')

    transposedConv2dLayer(3, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upD1') 
    dropoutLayer(0.1, 'Name', 'dropD1')
];

% Calculate Class Weights for Class Balancing as the number of background
% pixels outweights the number of flower pixels
% Author : MathWorks
% Source : “Count occurrence of pixel or box labels - MATLAB countEachLabel
% - MathWorks United Kingdom.” https://uk.mathworks.com/help/vision/ref/pixellabelimagedatastore.counteachlabel.html

% Calculating the inverse of the frequency of occurrence for each class
% Derived from the citated source
tbl = countEachLabel(pxds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;

finalLayers = [
    convolution2dLayer(1, numClasses, 'Name', 'finalConv')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'output','Classes',tbl.Name,'ClassWeights',classWeights);
];

% Create a graph of network layers, and connect the layers
% Author: Mathworks
% Source: “Graph of network layers for deep learning - MATLAB
% - MathWorks United Kingdom.” https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layergraph.html

% Intialise a layerGraph object
lgraph = layerGraph();

% Add layers from input_layer, encoder layers
lgraph = addLayers(lgraph, input_img);
lgraph = addLayers(lgraph, encoder1);
lgraph = addLayers(lgraph, encoder2);
lgraph = addLayers(lgraph, encoder3);
lgraph = addLayers(lgraph, encoder4);

% Add bottleneck layers
lgraph = addLayers(lgraph, bottleneck);

% Add decoder layers
lgraph = addLayers(lgraph, decoder4);
lgraph = addLayers(lgraph, decoder3);
lgraph = addLayers(lgraph, decoder2);
lgraph = addLayers(lgraph, decoder1);

% Add final layers
lgraph = addLayers(lgraph, finalLayers);

% Connect Input to Encoder 1
lgraph = connectLayers(lgraph,'conv_A','conv1_1');

% Connect layers in the encoder part
lgraph = connectLayers(lgraph, 'drop1', 'conv2_1');
lgraph = connectLayers(lgraph, 'drop2', 'conv3_1');
lgraph = connectLayers(lgraph, 'drop3', 'conv4_1');

% Connect layers to bottleneck
lgraph = connectLayers(lgraph, 'drop4', 'convB_1');

% Connect bottleneck to Decoder 1
lgraph = connectLayers(lgraph, 'dropB', 'convD1_1');

% Connect Decoder 2 to 4
lgraph = connectLayers(lgraph,'dropD1','convD2_1');
lgraph = connectLayers(lgraph, 'dropD2','convD3_1');
lgraph = connectLayers(lgraph, 'dropD3','convD4_1');

% Connect Decoder 4 to Final Layers
lgraph = connectLayers(lgraph,'dropD4','finalConv');

% Plot the layer graph to visualize the connections
plot(lgraph);

% Analyse Network to inspect for compilation errors
analyzeNetwork(lgraph);


% For this research's own network, adam optimiser would be used instead of
% sgdm, LearnRateDropPeriod= 10 compared to DeepLabv3+
% LearnRateDropPeriod=6 , L2Regularization has been set to 0.001

options = trainingOptions('adam', ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=10, ...
    L2Regularization=0.001, ...
    MaxEpochs=10, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ValidationData=dsVal, ...
    ValidationFrequency=10, ...
    Plots="training-progress");

% Train a segmentation network using the compiled network graph
net = trainNetwork(dsTrain,lgraph,options);

% Export the network file, when loaded into the workspace it would be
% initialised as net
save('segmentownnet.mat', 'net');

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

