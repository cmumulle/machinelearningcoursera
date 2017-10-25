function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% Calculate the mean
sum_mean = sum(X);                  % returns the sum over each colum
mu = 1/m * sum_mean';               % transpose sum_mean and divide by m

% Caclulate the variance
mu_sub = repmat(mu', m, 1);         % create a m x n matrix with means
err = X - mu_sub;                   % calculate the errors by feature and example
err_sqr = err .^2;                  % calculatte the squared errors
sum_err = sum(err_sqr);             % sum over all errors (by column)
sigma2 = 1/m * sum_err';            % sum over transposed errors and divide by m

% =============================================================

end
