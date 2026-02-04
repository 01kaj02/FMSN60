function [l]=lnl_spec(par,s,r)
% function [l]=lnl_spec(par,s,r)
% This symmetric distribution has mean zero and variance one (by design)
%
% -----Input----
% par parameters
% par(1) : tail parameter  c 0<=c<1 
%          c=0 gives a Gaussian distribution
%          in that case f does not impact the distribution
% par(2) : shape parameter f 0<f 
%          smoother density for higher f
%  s : scale parameter s 0<s
%          can be scalar or a vector with the same dimension as r  
% NOTE function return NaN for invalid parameters
% 
% r data vector
%
% ----Output----
% l loglikelihood
%
% (C) 2025 Magnus Wiktorsson
c=par(1);
f=par(2);
if (c<0)|(c>=1)|(f<=0)|(any(s<=0)) % check parameters
 l=NaN*r;   % return NaN if invalid region
else % otherwise compute likelihood
 fn=@(xx,c,f) (1+abs(xx).^f*(1-c))./(1+abs(xx).^f).^(1+c/f).*exp(-(xx./(1+abs(xx).^f).^(c/f)).^2/2)/sqrt(2*pi); % density of unscaled version
 v=@(c,f) 2*integral(@(u) u.*u.*fn(u,c,f),0,100); % calculate variance to scale it so that it is s^2
 lfn=@(xx,c,f) log(1+abs(xx).^f*(1-c))-(1+c/f)*log(1+abs(xx).^f)-(xx./(1+abs(xx).^f).^(c/f)).^2/2-log(2*pi)/2; % log likelihood of unscaled version
 sf=sqrt(v(c,f)); % find standard deviation to scale correctly 
 l=-log(s/sf)+lfn(r./(s/sf),c,f); % fix scale factor and compute final log likelihood
end
