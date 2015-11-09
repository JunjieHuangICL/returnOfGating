function W = getNetworkWeights(theta)
% returns the network weights in the struct 'theta' as one vector
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

idx = 0;
for i=1:length(theta)
    if isfield(theta{i},'W')
        idx = idx+numel(theta{i}.W);
    end
    if isfield(theta{i},'b')
        idx = idx+numel(theta{i}.b);
    end
end
W=zeros(idx,1);
idx=0;
for i=1:length(theta)
    if isfield(theta{i},'W')
        W(idx+[1:numel(theta{i}.W)]) = theta{i}.W(:);
        idx = idx+numel(theta{i}.W);
    end
    if isfield(theta{i},'b')
        W(idx+[1:numel(theta{i}.b)]) = theta{i}.b(:);
        idx = idx+numel(theta{i}.b);
    end
end
