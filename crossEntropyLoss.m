function [E,dE] = crossEntropyLoss(W,net,X,P)
% computes the cross entropy loss and its gradient for a given network
% W - the current weights of the network
% net - the network
% X - the input
% P - the refernce posterior probability
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%


% map vector W to cell array theta
net.theta = setNetworkWeights(W,net.theta);

% calc loss and gradient
[logQ,lin] = net.forward(net,permute(X,[1,3,4,5,2]));
logQ=permute(bsxfun(@minus,logQ,logsumexp(logQ)),[1,5,2,3,4]);
E = -mean(sum(P.*logQ));
if (nargout==1), return; end
delta = permute((exp(logQ)-P)/size(logQ,2),[1,5,2,3,4]);
dtheta = net.backward(net,lin,delta);

% map cell array dtheta to vector
dE = getNetworkWeights(dtheta);