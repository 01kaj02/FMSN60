%% Parameter estimation in SDE:s
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

addpath('/Users/khw/Documents/År 5/Finansiell Statistik/Labbar/Lab3')

load cirdata.mat
load cklsdata.mat


%% 3.1 Parameter estimation in stochastic differential equations

theta0 = mean(cirdata);
[xoutCIR,lOutCIR,CovMCIR] = MLmax(@lnL_CIR,[0.1 theta0 0.1],[cirdata]);

se = sqrt(diag(CovMCIR));
CI_low  = xoutCIR - 1.96*se';
CI_high = xoutCIR + 1.96*se';

fprintf('Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(1),  CI_low(1), CI_high(1));
fprintf('Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(2),  CI_low(2), CI_high(2));
fprintf('Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(3),  CI_low(3), CI_high(3));

theta0ckls = mean(cklsdata);
[xoutCKLS,lOutCKLS,CovMCKLS] = MLmax(@lnL_CKLS,[0.1 theta0ckls 0.1 0.1],[cklsdata]);

se = sqrt(diag(CovMCKLS));
CI_low  = xoutCKLS - 1.96*se';
CI_high = xoutCKLS + 1.96*se';

fprintf('Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(1),  CI_low(1), CI_high(1));
fprintf('Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(2),  CI_low(2), CI_high(2));
fprintf('Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(3),  CI_low(3), CI_high(3));
fprintf('Estimated gamma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(4),  CI_low(4), CI_high(4));

%% The GMM for CIR


par_hat_GMM = fminunc(@(params) GMM(params, cirdata), [0.5 theta0 0.1]);

[~, g] = GMM(par_hat_GMM, cirdata);
CovGMM = cov(g); 
par_hat_GMM2 = fminunc(@(params) GMM2(params, cirdata, inv(cov(g))), [0.3 0.1 0.1]);

% A = G' * CovGMM * G;
% B = G' * CovGMM * S * CovGMM * G;
% 
% CovGMM = inv(A) * B * inv(A);

se = sqrt(diag(CovGMM));
CI_low = par_hat_GMM2 - 1.96 * se';
CI_high = par_hat_GMM2 + 1.96 * se';
  
fprintf('Previous estimate of kappa = %8.4f | Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(1), par_hat_GMM2(1),  CI_low(1), CI_high(1));
fprintf('Previous estimate of theta = %8.4f | Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(2), par_hat_GMM2(2),  CI_low(2), CI_high(2));
fprintf('Previous estimate of sigma = %8.4f | Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(3), par_hat_GMM2(3),  CI_low(3), CI_high(3));

%% The GMM for CKLS 

theta0 = mean(cklsdata);
par_hat_GMMCKLS = fminunc(@(params) GMMCKLS(params, cklsdata), [0.1 theta0ckls 0.1 0.1]);

[~, g] = GMMCKLS(par_hat_GMMCKLS, cklsdata);
CovGMMCKLS = cov(g); 
par_hat_GMMCKLS2 = fminunc(@(params) GMM2CKLS(params, cirdata, inv(cov(g))), [0.1 theta0ckls 0.1 0.1]);

% A = G' * CovGMMCKLS * G;
% B = G' * CovGMMCKLS * S * CovGMMCKLS * G;
% 
% CovGMMCKLS = inv(A) * B * inv(A);

se      = sqrt(diag(CovGMMCKLS)); 
CI_low  = par_hat_GMMCKLS2 - 1.96 * se';
CI_high = par_hat_GMMCKLS2 + 1.96 * se';
  
fprintf('Previous estimate of kappa = %8.4f | Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(1), par_hat_GMMCKLS2(1),  CI_low(1), CI_high(1));
fprintf('Previous estimate of theta = %8.4f | Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(2), par_hat_GMMCKLS2(2),  CI_low(2), CI_high(2));
fprintf('Previous estimate of sigma = %8.4f | Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(3), par_hat_GMMCKLS2(3),  CI_low(3), CI_high(3));
fprintf('Previous estimate of gamma = %8.4f | Estimated gamma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCKLS(4), par_hat_GMMCKLS2(4), CI_low(4), CI_high(4));


%% 3.2 Exact likelihood for the CIR model

plot(cirdata);

dt = 1;

kappa = 0.1;
theta = mean(cirdata);
sigma = 0.1;

par = [log(kappa), log(theta), log(sigma)];

negloglik = @(p) -CIRloglik(p, cirdata, dt);


[par_hat,~, ~, ~, ~, hess_hat]  = fminunc(negloglik, par);
par_hat = exp(par_hat);

logcov = inv(hess_hat); % Delta method 
J = diag(par_hat);       
Cov = J * logcov * J.';  

se = sqrt(diag(Cov));

CI_low  = par_hat - 1.96*se';
CI_high = par_hat + 1.96*se';

fprintf('Kappa = %8.4f | Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(1), par_hat(1), CI_low(1), CI_high(1));
fprintf('Theta = %8.4f | Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(2), par_hat(2), CI_low(2), CI_high(2));
fprintf('Sigma = %8.4f | Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', xoutCIR(3), par_hat(3), CI_low(3), CI_high(3));


%% 4.1 Transformation of data for the CIR-model and the Shoji & Ozaki approximated likelihood 

dt = 1;

kappa = 0.1;
theta = mean(cirdata);
sigma = 0.1;

par = [log(kappa), log(theta), log(sigma)];

SO_negloglik = @(p) -SO_loglik_CIR(p, cirdata, dt);

[par_hat_SO,~, ~, ~, ~, hess_hat_SO]  = fminunc(SO_negloglik, par);

par_hat_SO = exp(par_hat_SO);

logcov_SO = inv(hess_hat_SO);
J = diag(par_hat_SO);       
Cov_SO = J * logcov_SO * J.';  

se = sqrt(diag(Cov_SO));

CI_low_SO  = par_hat_SO - 1.96*se';
CI_high_SO = par_hat_SO + 1.96*se';

fprintf('Previous estimate of kappa = %8.4f | Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat(1), par_hat_SO(1), CI_low_SO(1), CI_high_SO(1));
fprintf('Previous estimate of theta = %8.4f | Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat(2), par_hat_SO(2), CI_low_SO(2), CI_high_SO(2));
fprintf('Previous estimate of sigma = %8.4f | Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat(3), par_hat_SO(3), CI_low_SO(3), CI_high_SO(3));

%% 4.3 Exact Moments (GMM/EF)

par_hat_exactGMM = fminunc(@(params) exactGMM(params, cirdata), [0.2 theta0 0.1]);

[~, g] = exactGMM(par_hat_exactGMM, cirdata);
CovGMM = cov(g); 

se      = sqrt(diag(CovGMM));
CI_low  = par_hat_exactGMM - 1.96 * se';
CI_high = par_hat_exactGMM + 1.96 * se';
  
fprintf('Previous estimate of kappa = %8.4f | Estimated kappa = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat_GMM(1), par_hat_exactGMM(1),  CI_low(1), CI_high(1));
fprintf('Previous estimate of theta = %8.4f | Estimated theta = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat_GMM(2), par_hat_exactGMM(2),  CI_low(2), CI_high(2));
fprintf('Previous estimate of sigma = %8.4f | Estimated sigma = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', par_hat_GMM(3), par_hat_exactGMM(3),  CI_low(3), CI_high(3));


%% Functions


function l=lnL_CIR(theta,X)
Delta = 1;
k = theta(1);
thet = theta(2);
sigma = theta(3);
xn = X(1:end-1);
xn1 = X(2:end);
mu = xn+k*(thet-xn).*Delta;
sigma2 = (sigma.^2) .* xn * Delta;
l=-.5*log(2*pi*sigma2)-0.5*((xn1-mu).^2)./sigma2;
end


function l=lnL_CKLS(theta,X)
Delta = 1;
k = theta(1);
thet = theta(2);
sigma = theta(3);
gamma = theta(4);
xn = X(1:end-1);
xn1 = X(2:end);
mu = xn+k*(thet-xn).*Delta;
sigma2 = (sigma.^2) .* (xn.^(2*gamma)) .* Delta;
l=-.5*log(2*pi*sigma2)-0.5*((xn1-mu).^2)./sigma2;
end

function [g,G] = GMM(theta,X)
    k = theta(1);
    thet = theta(2);
    sigma = theta(3);

     if k <= 0 || thet <= 0 || sigma <= 0 
        g = 1e20;
        G = 0;
        return;
     end
    Delta = 1; 
    W = eye(3);
    xn = X(1:end-1);
    xn1 = X(2:end);
    g1 = xn1-xn - k.*(thet-xn).*Delta;
    g2 = xn1.^2 - ((xn + k.*(thet-xn).*Delta).^2 + sigma.^2.*xn.*Delta);
    g3 = xn1.^3 - ((xn + k.*(thet-xn).*Delta).^3 + 3.*((xn+k.*(thet-xn).*Delta).*sigma.^2.*xn.*Delta));
    g = [g1 g2 g3];
    G=g;
    g = mean(g)*W*mean(g)';
end


function g = GMM2(theta,X,W) 
    k = theta(1);
    thet = theta(2);
    sigma = theta(3);

    if k <= 0 || thet <= 0 || sigma <= 0 
        g = 1e20;
        return;
     end

    Delta = 1; 
    xn = X(1:end-1);
    xn1 = X(2:end);
    g1 = xn1-xn - k.*(thet-xn).*Delta;
    g2 = xn1.^2 - ((xn + k.*(thet-xn).*Delta).^2 + sigma^2.*xn.*Delta);
    g3 = xn1.^3 - ((xn + k.*(thet-xn).*Delta).^3 + 3.*((xn+k.*(thet-xn).*Delta).*sigma^2.*xn.*Delta));
    g = [g1 g2 g3];

    g = mean(g)*W*mean(g)';
end




function [J, g] = GMMCKLS(theta, X)
    % theta = [kappa, theta, sigma, gamma]

    kappa = theta(1);
    thet  = theta(2);
    sigma = theta(3);
    gamma = theta(4);
    
    if kappa <= 0 || thet <= 0 || sigma <= 0 || gamma <= 0 
        J = 1e20;   
        return;
    end

    Delta = 1;
    W = eye(4);    

    xn  = X(1:end-1);
    xn1 = X(2:end);

    g1 = xn1 - xn - kappa*(thet - xn)*Delta;
    g2 = xn1.^2 - (xn - kappa*(thet-xn)*Delta).^2 - sigma.^2*xn.^(2*gamma)*Delta;
    g3 = g1.*xn;
    g4 = g2.*xn;

    g = [g1 g2 g3 g4];      

    J = mean(g) * W * mean(g)';
end

function g = GMM2CKLS(theta,X,W) 
    kappa = theta(1);
    thet = theta(2);
    sigma = theta(3);
    gamma = theta(4);

    if kappa <= 0 || thet <= 0 || sigma <= 0 || gamma <= 0 
        g = 1e20;
        return;
     end

    Delta = 1; 
    xn = X(1:end-1);
    xn1 = X(2:end);
    g1 = xn1 - xn - kappa.*(thet - xn).*Delta;
    g2 = xn1.^2 - (xn - kappa*(thet-xn)*Delta).^2 - sigma.^2.*xn.^(2*gamma)*Delta;
    g3 = g1.*xn;
    g4 = g2.*xn;

    g = [g1 g2 g3 g4];

    g = mean(g)*W*mean(g)';
end

function logL = CIRloglik(par, r, dt)
% par  = [ln(kappa), ln(theta), ln(sigma)]

    kappa = exp(par(1));
    theta = exp(par(2));
    sigma = exp(par(3));

    r1 = r(1:end-1);   % x_k-1
    r2 = r(2:end);     % x_k

    epsr = 1e-12;
    r1 = max(r1, epsr);
    r2 = max(r2, epsr);

    c = 2 * kappa ./ (sigma^2*(1 - exp(-kappa*dt)));
    u = c .* r1 .* exp(-kappa*dt);
    v = c .* r2;
    q = 2 * kappa * theta / sigma^2 - 1;

    z = 2 * sqrt(u .* v);
    I = besseli(q, z);

    logp = log(c) -u-v +(q/2) .* (log(v) - log(u))  + log(I); % log p = log(c) - u - v + (q/2)*(log(v) - log(u)) + log(I_q(2*sqrt(uv)))
  
    logL = sum(logp); 
end





function logL = SO_loglik_CIR(par, r, dt)
% par = [log(kappa), log(theta), log(sigma)]

    kappa = exp(par(1));
    theta = exp(par(2));
    sigma = exp(par(3));

    y = 2 * sqrt(r);

    y1 = y(1:end-1);                   % y_{k-1}
    y2 = y(2:end);                     % y_k

    A = kappa*theta-sigma^2/4;

    a =  2*A./y1 - 0.5*kappa.* y1;
    b = -2*A./(y1.^2) - 0.5*kappa;
    c =  4*A./(y1.^3);

    epsb = 1e-12;
    b(abs(b) < epsb) = epsb;

    K = exp(b * dt) - 1;

    m = y1 + (a./b) .*K+(sigma^2 .* c ./ (2 * b.^2)) .* (K - b * dt);

    v = (sigma^2./(2*b)).* (exp(2*b*dt)-1);

    epsv = 1e-12;
    v = max(v, epsv);

    logp = -0.5*log(2*pi*v)-(y2-m).^2 ./(2*v);
    logL = sum(logp);
end



function [J,g] = exactGMM(theta, X)
    % theta = [kappa, theta, sigma]

    kappa = theta(1);
    thet  = theta(2);
    sigma = theta(3);

    if kappa <= 0 || thet <= 0 || sigma <= 0 
        J = 1e20;   
        return;
    end

    Delta = 1;

    xn  = X(1:end-1);                 % X_t
    xn1 = X(2:end);                   % X_{t+1}
    T   = numel(xn);

    a  = exp(-kappa * Delta);
    m1 = thet + (xn - thet) .* a; 

    g = [m1 0 0];
    J = 0;
    
end 


