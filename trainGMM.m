function [model] = trainGMM(model0,getData,param,filename)
% model0 - initial model or number of components for random init
% getData - function handle to get a random minibatch
% param - struct of paramteres:
%   p - patch size
%   mbnumber - number of minibatches/iterations
%   reg - regularization of covariance
%   draw - draw results
%
% Based on code by Daniel Zoran.
%
% Used in the paper: 
% "The Return of the Gating Network: Combining Generative Models and Discriminative 
% Training in Natural Image Priors" by Dan Rosenbaum and Yair Weiss
% 
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

draw = isfield(param,'draw') && param.draw;
llh_vec = ones(1,param.mbnumber);

% init
if (isstruct(model0))
  model = model0;
else  % random init
  model.p = param.p;
  model.ncomps = model0;
  X = getData();
  M = size(X,2);
  logRho = -20*rand(M,model.ncomps);
  llh = logsumexp(logRho,2);
  [model.mixweights, model.covs] = mStep(X,logRho,llh,param.reg);
end
model.param=param;

% EM iterations
for t=1:param.mbnumber
  tic;
  eta = 0.3;
 
  % get a new mini-batch
  X = getData();
 
  % E step
  [llh,logRho] = eStep(X,model);
  llh_vec(t)=mean(llh)/size(X,1)/log(2);
  fprintf('llh of GMM: %f\n', llh_vec(t));
  if (draw)
      figure(1); clf;
      subplot(2,1,1); hold on; xlabel('loglikelihood');
      plot(llh_vec(1:t),'bo-'); 
      subplot(2,1,2); hold on; xlabel('sorted mixing weights');
      plot(sort(model.mixweights,'descend'),'bo-'); 
      drawnow;
  end
 
  % M step
  [mixweights, covs] = mStep(X,logRho,llh,param.reg);
  model.mixweights = model.mixweights*(1-eta) + eta*mixweights;
  model.covs = model.covs*(1-eta) + eta*covs;
 
  toc;
  fprintf('iterarion %d of %d\n',t,param.mbnumber);
  if (exist('filename','var'))
    save([filename,'.mat'],'model','t','llh_vec');
  end
end


function [mixweights, covs] = mStep(X,logRho,llh,reg)
% compute the covariance of each component
m = size(X,2);
mixweights = exp(logsumexp(bsxfun(@minus,logRho,llh),1))/m;
covs = zeros(size(X,1),size(X,1),size(logRho,2));
parfor k = 1:size(logRho,2)
  logw = logRho(:,k)-llh;
  W = spdiags(exp(logw-logsumexp(logw)),0,m,m);
  Sig = (X*W*X');
  % covariance regularization
  [V,D] = svd(Sig);
  d = max(diag(D),reg);
  covs(:,:,k) = V*diag(d)*V';
end

 
function [llh,logRho] = eStep(X, model)
% compute assignemt probabilites, i.e. responsibilities. 
Sigma = model.covs;
logRho1 = zeros(size(X,2),model.ncomps);
parfor i = 1:model.ncomps
  logRho1(:,i) = loggausspdf(X,Sigma(:,:,i));
end
logRho = bsxfun(@plus,logRho1,log(model.mixweights));
llh = logsumexp(logRho,2);
