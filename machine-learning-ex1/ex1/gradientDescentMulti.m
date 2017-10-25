function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

numCol = columns(X);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    m = length(y);                      % number of training examples
    delta = zeros(numCol,1);                 % initialize delta
    
    prediction = X * theta;             % hypothesis for each training example
    difference = prediction - y;        % difference of hypotesis and actuals
    
    delta = 1/m * X' * difference;      % calculate delta to adjust theta
    
    theta = theta - alpha * delta;      % adjust theta


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
