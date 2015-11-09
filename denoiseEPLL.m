function [resI] = denoiseEPLL(Y, psize, patchMAP, lambdas, betas, stride)
% denoise the image 'Y' by minimizing the EPLL cost:
% J(X) = - sum_i logPr(X_i) + \lambda/2 ||X-Y||^2 
% where X is the image Y is the given noisy image and X_i are all the image patches.
%
% The cost is minimized by minimizing the following relaxation (penalty method):
% J(X,Z_i) = -sum_i logPr(Z_i) + lambda/2 ||X-Y||^2 + beta/2 sum_i||X_i-Z_i||^2
% where Z_i are allowed to be independent of each other but are gradually
% forced to equal X_i by increasing beta.
%
% In each iteration the problem solved by 
% (1) minimizing for each Z_i: -logPr(Z_i) + beta/2 ||X_i-Z_i||^2
%     (using the given patchMAP function handle).
% (2) minimize for X: lambda/2 ||X-Y||^2 + beta/2 sum_i||X_i-Z_i||^2
%     (by averaging all Z_i and Y).
% 
% input:
% Y - the noisy image
% psize - the patch size
% patchMAP - a patch denoising function handle, performing (approximate) MAP.
% lambdas - lambda in the cost (could be more than one value to allow it to
%           change through the iterations.
% betas - beta values for the different iterations (determins the # of iterations).
% stride - if bigger than 1, only a subset of patches are used (default=1).
%
% Used in the paper: 
% "The Return of the Gating Network: Combining Generative Models and Discriminative 
% Training in Natural Image Priors" by Dan Rosenbaum and Yair Weiss
% 
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

if ~exist('stride','var'), stride=1; end

% init im2col/col2im handles
[i2c,c2i] = fastIm2ColStrideHandle(size(Y),psize,stride,min(stride^2,length(betas)));

% init data
pp = prod(psize);
Y = Y - 0.5;
X = Y;


for i=1:length(betas)
    beta = betas(i);  
    lambda = lambdas(min(i,length(lambdas)));
    
    % minimize (1)
    Zi = patchMAP(i2c(X,i),1/beta); 
    
    % minimize (2)
    X = (pp*beta*c2i(Zi,i) + lambda*Y)./(pp*beta + lambda);
    
end

resI =  X + 0.5;
