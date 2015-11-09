function [X0,mu] = rmmean(X,dim)
% remove the mean (DC component) of X along dimension of dim
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%
if ~exist('dim','var');
    dim=1;
end
mu = mean(X,dim);
X0 = bsxfun(@minus,X,mu);
