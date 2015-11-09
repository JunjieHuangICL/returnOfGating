function [Xhat,logPXZ,cost,PSNR]=MAPGMM(Xhat0,Y,GMM,noiseVar,param,X)
% Computes the MAP estimation of X|Y where X is assmued to come from a GMM
% distribution and Y|X is an isotropic Gaussian. 
% The optimization is done using the EM algorithm.
% 
% Inputs:
% Xhat0 - initial quess (d by M where d is the dimension of the data and M
%                        is the number of data points)
% Y - the noisy data (d by M)
% GMM - the Gaussian Mixture Model prior for X. If it cotains a 'net' field
%       then it is used for the E step (calculating the responsibilities).
% noiseVar - the variance of Y|X
% param - parameter struct containing the following fields:
%         T      - number of iterations
%         hardEM - compute a hard version of EM (faster) 
%         calcCost - calc the full cost at each step
% X - the ground truth data, used to calculate the PSNR in each iteration.
%
% Outputs:
% Xhat - the MAP estimate
% logPXZ - the responsibilities of the last EM itration
% cost - the cost at each iteration (if param.calcCost=1)
% PSNR - the PSNR at each iteration (if X is given)
%
% Used in the paper: 
% "The Return of the Gating Network: Combining Generative Models and Discriminative 
% Training in Natural Image Priors" by Dan Rosenbaum and Yair Weiss
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

Xhat = Xhat0;
M = size(Xhat,2);
logPX = zeros(param.T+1,M);
cost = zeros(1,param.T+1);
PSNR = zeros(1,param.T+1);
prevks=-ones(1,size(Xhat,2));
if (~param.calcCost), numIter=param.T; else numIter=param.T+1; end

% prepare filters
A = zeros(size(GMM.covs));
for i=1:GMM.ncomps
    A(:,:,i) = (GMM.covs(:,:,i)+noiseVar*eye(size(Y,1)))\GMM.covs(:,:,i);
end

% EM itrations
for t=1:numIter
    % E step
    [logPXZ,logPX(t,:)] = calcResp(Xhat,GMM,param.calcCost);
    cost(t) = - mean(logPX(t,:),2) + 0.5/noiseVar*sum((Xhat(:)-Y(:)).^2)/M;
    if exist('X','var'), PSNR(t) = 20*log10(1/std(Xhat(:)-X(:))); end
    if (t>param.T), break;  end
    
    % M step
    if (param.hardEM)
        [Xhat,prevks] = compHardFilter(Y,A,logPXZ,prevks,Xhat);
    else
        Xhat = compSoftFilter(Y,A,logPXZ);
    end
end

function [logPXZ,logPX]=calcResp(X,GMM,calcCost)

% full posterior calculation
if (calcCost || (~isfield(GMM,'net')))
    logPXZ = zeros(GMM.ncomps,size(X,2));
    for i=1:GMM.ncomps
        logPXZ(i,:) = log(GMM.mixweights(i)) + loggausspdf(X,GMM.covs(:,:,i));
    end
    logPX = logsumexp(logPXZ);
else
    logPX = 0;
end

% posterior according to gating network if available
if (isfield(GMM,'net'))
    logPXZ = permute(GMM.net.forward(GMM.net,permute(X,[1,3,4,5,2])),[1,5,2,3,4]);
end

function Xhat = compSoftFilter(Y,A,logPXZ)
% filter noisy patches by a weighted sum over all components 
pZgX = exp(bsxfun(@minus,logPXZ,logsumexp(logPXZ)));
Xhat = zeros(size(Y));
for i=1:size(covs,3)
    Xhat = Xhat + bsxfun(@times,A(:,:,i)*Y,pZgX(i,:));
end

function [Xhat,ks] = compHardFilter(Y,A,logPXZ,prevks,prevXhat)
% filter noisy patches according to most probable component 
Xhat = zeros(size(Y));
[~,ks] = max(logPXZ);
Xhat(:,ks==prevks) = prevXhat(:,ks==prevks);
for i=1:size(A,3)
    idx = (ks==i & prevks~=i);
    Xhat(:,idx) = A(:,:,i)*Y(:,idx);
end

