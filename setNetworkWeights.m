function theta = setNetworkWeights(W,theta)
% replaces the weights in the struct 'theta' by the weights given in
% the vector W.
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

idx=0;
for i=1:length(theta)
    if isfield(theta{i},'W')
        theta{i}.W = reshape(W(idx+[1:numel(theta{i}.W)]),size(theta{i}.W));
        idx = idx+numel(theta{i}.W);
    end
    if isfield(theta{i},'b')
        theta{i}.b = reshape(W(idx+[1:numel(theta{i}.b)]),size(theta{i}.b));
        idx = idx+numel(theta{i}.b);
    end
end
