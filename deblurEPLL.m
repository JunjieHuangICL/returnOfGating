function [resI] = deblurEPLL(Y, psize, patchMAP, K, lambdas, betas,stride)
% Deblur the image 'Y' by minimizing the EPLL cost:
% J(X) = - sum_i logPr(X_i) + \lambda/2 ||A*X-Y||^2 
% where X is the image, Y is the given blurred image, A is a matrix
% implementation of the blurring (convolving K), and X_i are all the image patches.
%
% The cost is minimized by minimizing the following relaxation (penalty method):
% J(X,Z_i) = -sum_i logPr(Z_i) + lambda/2 ||A*X-Y||^2 + beta/2 sum_i||X_i-Z_i||^2
% where Z_i are allowed to be independent of each other but are gradually
% forced to equal X_i by increasing beta.
%
% In each iteration the problem solved by 
% (1) minimizing for each Z_i: -logPr(Z_i) + beta/2 ||X_i-Z_i||^2
%     (using the given patchMAP function handle).
% (2) minimize for X: lambda/2 ||X-Y||^2 + beta/2 sum_i||A*X_i-Z_i||^2
%     (by averaging all Z_i and Y).
% 
% input:
% Y - the noisy image
% psize - the patch size
% patchMAP - a patch denoising function handle, performing (approximate) MAP.
% K - the blur kernel
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
[i2c,c2i,~,pcount] = fastIm2ColStrideHandle(size(Y),psize,stride,min(stride^2,length(betas)));

% init data
Y = Y - 0.5;
X = Y;

for i=1:length(betas)
    beta = betas(i);  
    lambda = lambdas(min(i,length(lambdas)));
    
    % minimize (1)
    Zi = patchMAP(i2c(X,i),1/beta);
    
    % minimize (2)
    X = (mixI(c2i(Zi,i),Y,K,lambda,beta,pcount(i)));
end

resI =  X + 0.5;



% mix the average image with the blurred image
function cleanI=mixI(I1,noisyI,K,lambda,beta,pcount)

tt1 = noisyI(floor(size(K,1)/2)+1:end-floor(size(K,1)/2),floor(size(K,2)/2)+1:end-floor(size(K,2)/2));
tt1 = conv2(tt1,rot90(rot90(K)),'full');
tt2 = pcount.*I1;
[deco,~] = bicg(@(x,tflag) Afun(x,K,pcount,beta,lambda,size(noisyI)),(lambda*tt1(:) + (beta)*tt2(:)),1e-5,200);
cleanI = reshape(deco,size(noisyI));

% function to apply the corruption model (implemented efficiantly using
% convolutions instead of matrix multiplications)
function y = Afun(x,K,counts,beta,lambda,ss)
xx = reshape(x,ss);
tt = imfilter(xx,K,'conv','same');
tt = tt(floor(size(K,1)/2)+1:end-floor(size(K,1)/2),floor(size(K,2)/2)+1:end-floor(size(K,2)/2));
y = lambda*imfilter(tt,rot90(rot90(K)),'conv','full');
y = y + beta*counts.*xx;
y = y(:);
