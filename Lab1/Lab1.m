%% Parameter estimation and linear time-series analysis
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

load regdataN.mat
load regdatat.mat


%% 4.1 Linear Regression

XYN = [x yN];
XYT = [x yT];

[xoutN,logL,CovM] = MLmax(@lnL,[1 1 1],XYN); 
[xoutT,logLT,CovMT] = MLmax(@lnLT,[1 1 1 1],XYT); 



% Confidence interval - normal 
z = 1.96;  

thetaN = xoutN(:);                  
seN = sqrt(diag(CovM)); 
CI_N = [thetaN - z*seN, thetaN + z*seN];      


% Confidence interval - student t  
thetaT = xoutT(:);
seT = sqrt(diag(CovMT));

CI_T = [thetaT - z*seT, thetaT + z*seT];


disp(CI_N);
disp(CI_T);

% Comparison 
X = [ones(size(x)) x];

[LSN, CI_LSN] = regress(yN,X);
[LST, CI_LST] = regress(yT,X);


% LR-test

[xoutQ, logLQ] = MLmax(@lnLquad, [1 1 1 1], [x yN]);

LR = -2*(logL-logLQ);

disp( LR);

crit = chi2inv(0.95, 1);

if LR > crit
    disp('Reject H0: quadratic model significantly improves the fit.');
else
    disp('Fail to reject H0: linear model is sufficient.');
end

%% 4.2 Time Series Analysis
clear;
load prisdata.mat

%plot(p);

r = -log(p(1:end-1))./(T-t(1:end-1));
figure;
 plot(r);


rm=r-mean(r);
sacfplot(rm,30);
title('ACF');

y = rm(2:end);          %r_t
x = rm(1:end-1);        %r_{t-1}

%[LS] = regress(y,x);

% Residuals
[xout,logL,Cov] = MLmax(@lnLAR,[1 1],[x y]); 

phi = xout(1);

res = y - phi * x;
sacfplot(res, 30); 
title('ACF of AR(1) residuals');

% This is not sufficient so try an AR(2)-model instead

y2 = rm(3:end);          %r_t
x2 = rm(2:end-1);        %r_{t-1}
z2 = rm(1:end-2);        %r_{t-2}


[xout2, logL2] = MLmax(@lnL2, [1 1 1 ], [x2 y2 z2 ]);


phi = xout2(1);
phi2 = xout2(2);

res2 = y2 - phi * x2-phi2*z2;
sacfplot(res2, 30); 
title('ACF of AR(2) residuals');

% This looks a lot better but we will try AR(3) as well


y3 = rm(4:end);          %r_t
x3 = rm(3:end-1);        %r_{t-1}
z3 = rm(2:end-2);        %r_{t-2}
w3 = rm(1:end-3);        %r_{t-3}



[xout3, logL3] = MLmax(@lnL3, [1 1 1 1], [x3 y3 z3 w3]);


phi = xout3(1);
phi2 = xout3(2);
phi3 = xout3(3);

res3 = y3 - phi * x3-phi2*z3 -phi3*w3;
sacfplot(res3, 30); 
title('ACF of AR(3) residuals');

% This looks worse, but we do a likelihood ratio test. 

LR = -2*(logL2-logL3);

disp( LR);

crit = chi2inv(0.95, 1);

if LR > crit
    disp('Reject H0: quadratic model significantly improves the fit.');
else
    disp('Fail to reject H0: linear model is sufficient.');
end


 % Comparison
 th1 = arx(rm,1);
 present(th1)
 th2 = arx(rm,2);
 present(th2)


%% Function


function l=lnL(theta,X)
    x = X(:,1);
    y = X(:,2);

    beta0 = theta(1);
    beta1 = theta(2);
    sigma = theta(3);

    mu = beta0 + beta1*x;

    l = -0.5*log(2*pi*sigma^2) - 0.5*((y - mu).^2)/(sigma^2);
end 

function l=lnLT(theta,X)
    x = X(:,1);
    y = X(:,2);

    beta0 = theta(1);
    beta1 = theta(2);
    sigma = theta(3);
    v = theta(4);

    mu = beta0 + beta1*x;

    z = (y - mu).^2 ./ (v * sigma^2);

    l = gammaln((v+1)/2) - gammaln(v/2) - 0.5*log(pi*v) - log(sigma)- ((v+1)/2)*log(1 + z);

    l = l(:);
end 


function l=lnLquad(theta,X)
    x = X(:,1);
    y = X(:,2);

    beta0 = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    sigma = theta(4);

    mu = beta0 + beta1*x + beta2*x.^2;

    l = -0.5*log(2*pi*sigma^2) - 0.5*((y - mu).^2)/(sigma^2);
end 

function l=lnLAR(theta,X)
    x = X(:,1);
    y = X(:,2);

    phi = theta(1);
    sigma = theta(2);

    mu = phi*x;

    l = -0.5*log(2*pi*sigma^2) - 0.5*((y - mu).^2)/(sigma^2);
    l=l(:);
end 

function l=lnL2(theta,X)
    x = X(:,1);
    y = X(:,2);
    z = X(:,3);

    phi = theta(1);
    phi2 = theta(2);
    sigma = theta(3);

    mu = phi*x+phi2*z;

    l = -0.5*log(2*pi*sigma^2) - 0.5*((y - mu).^2)/(sigma^2);
    l=l(:);
end 

function l=lnL3(theta,X)
    x = X(:,1);
    y = X(:,2);
    z = X(:,3);
    w = X(:,4);

    phi = theta(1);
    phi2 = theta(2);
    phi3 = theta(3);
    sigma = theta(4);

    mu = phi*x+phi2*z+phi3*w;

    l = -0.5*log(2*pi*sigma^2) - 0.5*((y - mu).^2)/(sigma^2);
    l=l(:);
end 
