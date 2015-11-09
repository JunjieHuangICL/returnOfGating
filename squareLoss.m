function [L,dL] = squareLoss(W,net,data)
% computes the square loss and its gradient for a given network
% W - the current weights of the network
% net - the network
% data - containing the data x, and the MAP latent variables y
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%


M = size(data.x,2);

% map vector W to cell array theta
net.theta = setNetworkWeights(W,net.theta);

% calc current response y
[Y,lin] = net.forward(net,permute(data.x,[1,3,4,5,2]));
Y = permute(Y,[1,5,2,3,4]);

% square loss
L = 0.5*sum((Y(:)-data.y(:)).^2)/M;

if (nargout<2), return; end;

% gradient
dy = (Y-data.y)/M;
    
% calc gradient of network weights
dtheta = net.backward(net,lin,dy);
    
% map cell array dtheta to vector
dL = getNetworkWeights(dtheta);
    