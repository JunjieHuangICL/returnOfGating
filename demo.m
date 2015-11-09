%% load pre-trained models (see demoTrain.m for training models)
GMM = getfield(load('GMM200.mat'),'model');
GMM.net = getfield(load('gating100.mat'),'net');

%% image denoising

% load image 
origI = mean(double(imread('123074.jpg')),3)/255;

% add noise
noiseVar = (25/255)^2;
noisyI = origI + sqrt(noiseVar)*randn(size(origI));

% denoise params
stride = 3; % change for speed/accuracy tradeoff (might require update of lambda)
lambdas = GMM.p^2/noiseVar*[0.7,0.6,0.5,0.4];
betas = [1,5,10,50]/noiseVar;
% GMM patch denoising function handle
patchMAP = @(Zi,noiseVar) MAPGMM(Zi,Zi,GMM,noiseVar,struct('T',3, 'hardEM',1, 'calcCost',0)); 


% denoise 
tic;
resI = denoiseEPLL(noisyI,[GMM.p,GMM.p],patchMAP,lambdas,betas,stride); 
t=toc;

PSNR = 20*log10(1./std2(resI-origI));
fprintf('denoising with gating network: time %f PSNR %f\n',t,PSNR);
figure(1); imshow(origI); title('original');
figure(2); imshow(noisyI); title('noisy');
figure(3); imshow(resI); title('denoised');

%% image deblurring

% load image
origI = mean(double(imread('123074.jpg')),3)/255;

% blur the image
noiseVar = (2.5/255)^2;
K = fspecial('motion',10,45);
y = conv2(origI, K, 'valid');
y = y + sqrt(noiseVar)*randn(size(y));
y = double(uint8(y .* 255))./255;
ks = floor((size(K, 1) - 1)/2);
y = padarray(y, [1 1]*ks, 'replicate', 'both');
for aa=1:4, y = edgetaper(y, K); end
blurI = y;
    
% deblur params
stride = 3;
lambdas = 0.2*GMM.p^2/noiseVar;
betas = 50*[1,2,4,8,16,32,64];
patchMAP = @(Zi,noiseVar) MAPGMM(Zi,Zi,GMM,noiseVar,struct('T',3, 'hardEM',1, 'calcCost',0)); 
%stride=1; lambdas = GMM.p^2/noiseVar;


% deblur
tic;
resI = deblurEPLL(blurI, [GMM.p,GMM.p], patchMAP, K, lambdas, betas, stride);
t=toc;

PSNR = 20*log10(1./std2(resI-origI));
fprintf('denoising with gating network: time %f PSNR %f\n',t,PSNR);
figure(1); imshow(origI); title('original');
figure(2); imshow(blurI); title('blurred');
figure(3); imshow(resI); title('deblurred');

