function b = crosscorr(y,u,p)

crosscorr = xcorr(y,u,p);
%fprintf('size(crosscorr) is %s\n', mat2str(size(crosscorr)))
crosscorr = crosscorr(p+1:end);
b = crosscorr;