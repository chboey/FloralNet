%-------------------------------------------------------------------
% Custom model loss function: Normalised cross entropy that handles null
% values
% Author : MathWorks
% Source : MathWorks, “Semantic segmentation using Deep Learning - MATLAB & Simulink - MathWorks United Kingdom.” 
%          https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
function loss = modelLoss(Y,T) 
  mask = ~isnan(T);
  targets(isnan(T)) = 0;
  loss = crossentropy(Y,T,Mask=mask,NormalizationFactor="mask-included"); 
end
