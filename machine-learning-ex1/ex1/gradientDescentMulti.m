function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
J_history = zeros(num_iters, 1);
temp = zeros(n,1);
diff_x = zeros(1,n);

for iter = 1:num_iters
  pred = X * theta;
  for i = 1:n
    diff_x(i) = (pred-y)' * X(:,i); % size((pred-y) .* X(:,i)) = m*1
    temp(i,1) = theta(i,1) - ((alpha/m)*diff_x(i)); % size(sum(diff_x)) = 1*1
  end
  for i = 1:n
    theta(i,1) = temp(i,1);
  end
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
