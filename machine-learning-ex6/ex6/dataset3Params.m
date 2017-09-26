function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

cvec_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
low_error = 1000000000;

for i = 1:length(cvec_vec)
  for j = 1:length(sigma_vec)
    c_try = cvec_vec(i);
    sigma_try = sigma_vec(j);
    model= svmTrain(X, y, c_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
    predictions = svmPredict(model, Xval); 
    perror = mean(double(predictions ~= yval));
    if(perror < low_error)
      low_error = perror;
      C = c_try;
      sigma = sigma_try;
    end
  end
end

 





% =========================================================================

end
