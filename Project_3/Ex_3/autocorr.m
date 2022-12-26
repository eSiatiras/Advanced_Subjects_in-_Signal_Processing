function a = autocorr(u,p)
acorr = xcorr(u,p);
%fprintf('size(acorr) is %s\n', mat2str(size(acorr)))
acorr = acorr(p+1:end);
a = acorr;
