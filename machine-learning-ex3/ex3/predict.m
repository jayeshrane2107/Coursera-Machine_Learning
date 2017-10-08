function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1) X];
%fprintf('X = %d,%d\n',size(X));  5000*401

z1 = X*Theta1';
hx1 = sigmoid(z1);
%fprintf('hx1 = %d,%d\n',size(hx1));  5000*25
hx1 = [ones(size(hx1,1),1) hx1];
%fprintf('hx1 = %d,%d\n',size(hx1));  5000*26

z2 = hx1*Theta2';
hx2 = sigmoid(z2);
%fprintf('hx2 = %d,%d\n',size(hx2));  5000*10

[value, index] = max(hx2, [], 2);
p=index;

% =========================================================================


end
