function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


%h = transpose(theta).*X;
%h=sum(h,2);
h = sigmoid(X*theta);


temp = theta;
temp(1)=0;
temp=temp.^2;
temp= sum(temp,1);



J = sum((-y.*log(h)) - (1-y).*(log(1-h)))/m ;
J = J +(lambda*temp)/(2*m);

g_temp = theta;
g_temp(1)=0;

grad = grad + sum(transpose(h-y).*transpose(X),2)/m + (lambda*g_temp)/m;












% =============================================================

grad = grad(:);

end
