function [X,Xclean] = loadImagePatches(imdir,psize,M,noiseVar)
% load M patches from random locations in all the .jpg images located in
% imdir.
% X has (psize(1)*psize(2)) rows and M columns.
%
% If psize is a matrix than patches of different sizes (according to the 
% rows of psize) around the same centre are extracted and resized to 
% psize(1,[1,2]).
% In this case X has a 3rd dimension of size(psize,1).
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

if ~exist('noiseVar','var'), noiseVar=0; end;

a = dir([imdir,'*.jpg']);
MperIm = ceil(M/length(a));

xminus = floor((psize(:,2)-1)/2);
xplus = floor((psize(:,2))/2);
yminus = floor((psize(:,1)-1)/2);
yplus = floor((psize(:,1))/2);

X = zeros(psize(1,1)*psize(1,2),size(psize,1),length(a),MperIm);
if (nargout>1)
    Xclean = zeros(psize(1,1)*psize(1,2),1,length(a),MperIm);
end
for i=1:length(a)
    imClean = single(mean(imread([imdir,a(i).name]),3));
    im = imClean + sqrt(noiseVar)*randn(size(imClean));
    imsize = size(im);
    x = randi([1+max(xminus),imsize(2)-max(xplus)],MperIm);
    y = randi([1+max(yminus),imsize(1)-max(yplus)],MperIm);
    
    for pi=1:MperIm
        
        for li=1:size(psize,1) % get different levels for all patches
            patch = im(y(pi)-yminus(li):y(pi)+yplus(li),x(pi)-xminus(li):x(pi)+xplus(li));
            if (li>1), patch = imresize(patch,psize(1,:),'bilinear'); end                
            X(:,li,i,pi) = patch(:);
            if (li==1 && nargout>1)
                patch = imClean(y(pi)-yminus(li):y(pi)+yplus(li),x(pi)-xminus(li):x(pi)+xplus(li));
                Xclean(:,1,i,pi) = patch(:);
            end
        end
        
    end
end

X = permute(reshape(X,[psize(1,1)*psize(1,2),size(psize,1),length(a)*MperIm]),[1,3,2]);

rp=randperm(size(X,2),M);
X = X(:,rp,:);

if (nargout>1)
    Xclean = permute(reshape(Xclean,[psize(1,1)*psize(1,2),1,length(a)*MperIm]),[1,3,2]);
    Xclean = Xclean(:,rp,:);
end


