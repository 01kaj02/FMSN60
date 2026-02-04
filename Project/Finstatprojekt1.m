%% Financial Statistics Project Part 1
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

addpath('/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/')


 %% 1: Intra-day and daily data using standard GARCH-family models and GARCH-x models for volatility, track A
clear;
load('VOLV0A.mat') % 2003-01-02-2003-12-12 

close = [VOLVOA.close]'; 

r = log(close(2:end))-log(close(1:end-1)); % daily log returns
date = [VOLVOA.date]';    
date = datetime(date, 'ConvertFrom', 'yyyymmdd');
pred_dates = date(2:end);     
pred_dates = pred_dates(2:end);

n = length(r);
z = norminv(0.975);  

figure;
plot(date(2:end), r);
xlabel('Date')
ylabel('Daily log return')
title('Volvo Log-Return from Closing Prices')


%% Checking GARCH order & checking the distribution
  
e2 = r.^2; % mean can be ignored
T  = length(e2);

maxLag = 110;

u = e2 - mean(e2); % subtracting mean for correlation calculation

c = xcorr(u, maxLag, 'biased'); % sample autocovariances
acf = c(maxLag+1:end) / c(maxLag+1);  % normalize by lag-0

lags = (0:maxLag)';

figure;
stem(lags, acf, 'filled'); grid on;
xlabel('Lag'); ylabel('ACF');
title('ACF of squared returns x_t^2');

hold on;
conf = 1.96/sqrt(T);
plot([0 maxLag], [conf conf], 'r--');
plot([0 maxLag], [-conf -conf], 'r--');
hold off;

pacf = zeros(maxLag,1);

for k = 1:maxLag
    R = toeplitz(acf(1:k));  %autocorr matrix
    rhs = acf(2:k+1);            
    phi = R \ rhs;             
    pacf(k) = phi(end);        
end

figure;
stem((1:maxLag)', pacf, 'filled'); grid on;
xlabel('Lag'); ylabel('PACF');
title('Manual PACF of squared returns x_t^2');

hold on;
plot([1 maxLag], [conf conf], 'r--');
plot([1 maxLag], [-conf -conf], 'r--');
hold off;

% Checking the distribution (normal)
figure;
histogram(r, 'Normalization','pdf','EdgeColor','none');
hold on;

mu_hat = mean(r);
sigma_hat = std(r);

x = linspace(min(r), max(r), 200);
plot(x, normpdf(x, mu_hat, sigma_hat), 'r', 'LineWidth', 1.5);

xlabel('Daily log return');
ylabel('Density');
title('Histogram of daily returns with fitted normal density');
legend('Empirical density','Normal fit');
grid on;

figure;
qqplot(r);
title('Normal Q–Q plot of daily returns');
grid on;

% Checking stationarity
[h_adf, p_adf] = adftest(r);
[h_kpss, p_kpss] = kpsstest(r);
[h_arch, p_arch] = archtest(r);



%%  Bi-Power & Quadratic Variation

% Calculating the bi-power variation and the quadratic variation of the log returns
Ndays = numel(VOLVOA);
BV = zeros(Ndays,1);
QV = zeros(Ndays,1);
closing_prices = zeros(Ndays,1);

for i = 1:Ndays
    returns = log(VOLVOA(i).price(2:end))- log(VOLVOA(i).price(1:end-1));
    mu = pi/2;
    BV(i) = mu.*sum(abs(returns(1:end-1)).*abs(returns(2:end)));
    QV(i) = sum(returns.^2);
    closing_prices(i) =  VOLVOA(i).close;
end

% Plot of the bi-power variation and the quadratic variation.
figure;
plot(BV);
title('Bi-power variation');
figure;
plot(QV);
title('Quadratic variation');



%% GARCH-X
input = QV;

alpha_init = 0.6;
beta_init = 0.1;
mu_init = mean(r);
omega_init = 0.1;
gamma_init = 0.5;
x0 = [omega_init alpha_init beta_init gamma_init mu_init];
[xout,~,~] = estimation2(x0,[r(2:end), input(1:end-2)]);



omega_hatx = xout(1);
alpha_hatx = xout(2);
beta_hatx = xout(3);
gamma_hatx = xout(4);
mu_hatx = xout(5);

fprintf('Estimated omega = %8.4f \n', omega_hatx);
fprintf('Estimated alpha = %8.4f \n', alpha_hatx);
fprintf('Estimated beta = %8.4f \n', beta_hatx);
fprintf('Estimated gamma = %8.4f \n', gamma_hatx);
fprintf('Estimated mu = %8.4f \n', mu_hatx);
epsilon_hatx = r - mu_hatx; % residuals from (constant) mean estimate  (demeaned)
h_t = zeros(n,1);

if alpha_hatx + beta_hatx < 1
    EX = mean(input(1:end-1));   %mean of exogenous regressor
    h_t(1) = (omega_hatx + gamma_hatx * EX) / (1 - alpha_hatx - beta_hatx); %  unconditional variance (works if stationary)
else
    h_t(1) = var(epsilon_hatx);
end

for i=2:n
    h_t(i) = omega_hatx + alpha_hatx*epsilon_hatx(i-1)^2 + beta_hatx*h_t(i-1) + gamma_hatx*input(i-1);
end 

sigma_tx = sqrt(h_t);
figure;
plot(date(2:end), sigma_tx);
xlabel('Date');
ylabel('Conditional volatility (sigma_t)');
title('Estimated GARCH-X(1,1) volatility');

% next estimate
h_forxn1 = omega_hatx + alpha_hatx * epsilon_hatx(n)^2 + beta_hatx  * h_t(n) + gamma_hatx * input(n);
mu_forx = mu_hatx; 
sigma_forxn1 = sqrt(h_forxn1);
est_CI_lowxn1  = mu_forx - z * sigma_forxn1;
est_CI_highxn1 = mu_forx + z * sigma_forxn1;

% alla estimates
h_fore_garchx   = zeros(n-1,1);
CI_low_garchx   = zeros(n-1,1);
CI_high_garchx  = zeros(n-1,1);
for t = 1:n-1
    h_fore_garchx(t) = omega_hatx + alpha_hatx * epsilon_hatx(t)^2 + beta_hatx  * h_t(t) + gamma_hatx * input(t);
    sigma_forex = sqrt(h_fore_garchx(t));
    
    CI_low_garchx(t)  = mu_forx - z * sigma_forex; % också normalfördelning 
    CI_high_garchx(t) = mu_forx + z * sigma_forex;
end

mu_line = mu_forx * ones(size(pred_dates)); 
r_realized = r(2:end); 

figure; hold on; grid on;
h1 = plot(pred_dates, r_realized, 'k-', 'LineWidth', 1);
h2 = plot(pred_dates, mu_line, 'r--', 'LineWidth', 1);
h3 = plot(pred_dates, CI_low_garchx,  'b--', 'LineWidth', 1.2);
h4 = plot(pred_dates, CI_high_garchx, 'b--', 'LineWidth', 1.2);
xlabel('Date');
ylabel('Log return');
title('1-step-ahead 95% prediction intervals with GARCH-X');
legend([h1, h2, h3, h4], ...
    {'Realized return', ...
        'Conditional mean (forecast)',...
        'GARCH-X 95% lower','GARCH-X 95% upper'}, ...
        'Location','best');
hold off;


%% Standard GARCH(1,1)

alpha_init = 0.6;
beta_init = 0.1;
mu_init = mean(r);
omega_init = (1 - alpha_init - beta_init) * var(r - mu_init); % reverse of unconditional variance


x0 = [mu_init; omega_init; alpha_init; beta_init];

[xout,logL,CovM]=MLmax(@garchLL,x0,r); 


mu_hat = xout(1);
omega_hat = xout(2);
alpha_hat = xout(3);
beta_hat = xout(4);

fprintf('Estimated mu = %8.4f \n', mu_hat);
fprintf('Estimated omega = %8.4f \n', omega_hat);
fprintf('Estimated alpha = %8.4f \n', alpha_hat);
fprintf('Estimated beta = %8.4f \n', beta_hat);

epsilon_hat = r - mu_hat; % residuals from (constant) mean estimate 

h_t = zeros(n,1);
if alpha_hat + beta_hat < 1
    h_t(1) = omega_hat / (1 - alpha_hat - beta_hat); %  unconditional variance
else
    h_t(1) = var(epsilon_hat); 
end


for i=2:n
    h_t(i) = omega_hat + alpha_hat*epsilon_hat(i-1)^2 + beta_hat*h_t(i-1); 
end 

sigma_t = sqrt(h_t);

figure; 
plot(date(2:end), sigma_t);
xlabel('Date');
ylabel('Conditional volatility (sigma_t)');
title('Estimated GARCH(1,1) volatility');


% forecast för nästa 
h_foren1 = omega_hat + alpha_hat * epsilon_hat(n)^2 + beta_hat  * h_t(n);
mu_fore = mu_hat;
sigma_foren1 = sqrt(h_foren1);

est_CI_lown1  = mu_fore - z * sigma_foren1;
est_CI_highn1 = mu_fore + z * sigma_foren1;

% alla estimates

h_fore_garch   = zeros(n-1,1);
CI_low_garch   = zeros(n-1,1);
CI_high_garch  = zeros(n-1,1);

for t = 1:n-1
    h_fore_garch(t) = omega_hat + alpha_hat * epsilon_hat(t)^2 + beta_hat  * h_t(t);
    sigma_fore = sqrt(h_fore_garch(t));
    
    CI_low_garch(t)  = mu_hat - z * sigma_fore; % också normalfördelning 
    CI_high_garch(t) = mu_hat + z * sigma_fore;
end


mu_line = mu_hat * ones(size(pred_dates));

r_realized = r(2:end);         

figure; hold on; grid on;

h1 = plot(pred_dates, r_realized, 'k-', 'LineWidth', 1);
h2 = plot(pred_dates, mu_line, 'r--', 'LineWidth', 1);

h3 = plot(pred_dates, CI_low_garch,  'b--', 'LineWidth', 1.2);
h4 = plot(pred_dates, CI_high_garch, 'b--', 'LineWidth', 1.2);

xlabel('Date');
ylabel('Log return');
title('1-step-ahead 95% prediction intervals with GARCH');
legend([h1, h2, h3, h4], ...
    {'Realized return', ...
        'Conditional mean (forecast)',...
        'GARCH 95% lower','GARCH 95% upper'}, ...
        'Location','best');
hold off;



%% Standard EGARCH 

alpha_inite = 0.05;
beta_inite = 0.9;
omega_inite = (1 - beta_inite) * log(var(r-mu_init)); % simplified reverse of unconditional variance


ex0 = [mu_init; omega_inite; alpha_inite; beta_inite];


[exout,elogL,eCovM]=MLmax(@EGARCHLL,ex0,r);

SE = sqrt(diag(eCovM));

eCI_low  = exout - 1.96*SE;
eCI_high = exout + 1.96*SE;

mu_hate = exout(1);
omega_hate = exout(2);
alpha_hate = exout(3);
beta_hate = exout(4);

fprintf('Estimated mu    = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', mu_hate, eCI_low(1), eCI_high(1));
fprintf('Estimated omega = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', omega_hate, eCI_low(2), eCI_high(2));
fprintf('Estimated alpha = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', alpha_hate, eCI_low(3), eCI_high(3));
fprintf('Estimated beta  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', beta_hate, eCI_low(4), eCI_high(4));

epsilon_hate = r - mu_hate;  % demean

logh_t = zeros(n,1);
 
logh_t(1) = log(var(epsilon_hate)); 

for t = 2:n
    h_prev = exp(logh_t(t-1));              
    w_prev = epsilon_hate(t-1) / sqrt(h_prev);

    logh_t(t) = omega_hate + alpha_hate*w_prev + beta_hate*logh_t(t-1);
end

h_te = exp(logh_t);      

sigma_te = sqrt(h_te);

figure;
plot(date(2:end), sigma_te);
xlabel('Date');
ylabel('Conditional volatility (sigma_t)');
title('Estimated EGARCH(1,1) volatility');


% prediction one step ahead 
h_T = h_te(n);                      
w_T = epsilon_hate(n) / sqrt(h_T);       
logh_fore_en1 = omega_hate + alpha_hate*w_T + beta_hate*logh_t(n);

h_fore_en1 = exp(logh_fore_en1);   

sigma_fore_en1 = sqrt(h_fore_en1);
mu_fore_e = mu_hate;

est_CI_low_en1  = mu_fore_e - z * sigma_fore_en1; % här antar vi normal
est_CI_high_en1 = mu_fore_e + z * sigma_fore_en1;

% prediction for all 

h_fore_eg   = zeros(n-1,1);
CI_low_eg   = zeros(n-1,1);
CI_high_eg  = zeros(n-1,1);

for t = 1:n-1
    h_t_eg = exp(logh_t(t));           
    
    w_t = epsilon_hate(t) / sqrt(h_t_eg);
    
    logh_fore = omega_hate + alpha_hate * w_t + beta_hate  * logh_t(t); 
    
    h_fore_eg(t) = exp(logh_fore);
    sigma_fore   = sqrt(h_fore_eg(t));
    
    CI_low_eg(t)  = mu_hate - z * sigma_fore; 
    CI_high_eg(t) = mu_hate + z * sigma_fore;
end

mu_linee = mu_hate*ones(length(pred_dates),1);

figure; hold on; grid on;

h1 = plot(pred_dates, r_realized, 'k-', 'LineWidth', 1);   

h2 = plot(pred_dates, mu_linee, 'r--', 'LineWidth', 1);   

h3 = plot(pred_dates, CI_low_eg,  'b--', 'LineWidth', 1.2);     
h4 = plot(pred_dates, CI_high_eg, 'b--', 'LineWidth', 1.2);      

xlabel('Date');
ylabel('Log return');
title('1-step-ahead 95% prediction intervals with EGARCH(1,1)');
legend([h1 h2 h3 h4],...
    {'Realized return', ...
        'Conditional mean (forecast)', ...
        'EGARCH 95% lower','EGARCH 95% upper'}, ...
        'Location','best');

hold off;

%% Christoffersen tests 

x95_2 = chi2inv(0.95, 2);
x95_1 = chi2inv(0.95, 1);

T = length(r_realized);

% GARCH-X 
I_garchx = zeros(size(r_realized));

for (i=1:T)
    if (r_realized(i) >= CI_low_garchx(i)) && (r_realized(i) <= CI_high_garchx(i))
    I_garchx(i) = 1;
    end
end 


% conditional coverage
p = 0.95; 

n1x = sum(I_garchx);         
n0x = T-n1x; 

L_px = (1-p)^n0x*p^n1x;


n00x = 0;
n01x = 0;
n10x = 0;
n11x = 0; 

for(i=1:T-1)
    if (I_garchx(i)== 0 && I_garchx(i+1)==0)
        n00x=n00x+1;
    elseif (I_garchx(i)== 0 && I_garchx(i+1)==1)
        n01x=n01x+1;
    elseif(I_garchx(i)== 1 && I_garchx(i+1)==0)
        n10x=n10x+1;
    else
        n11x=n11x+1;
    end 
end 

PI_hatx = [n00x/(n00x+n01x) n01x/(n00x+n01x);
    n10x/(n10x+n11x)  n11x/(n10x+n11x) ];

L_pix = ((PI_hatx(1,1))^n00x)*(PI_hatx(1,2)^n01x)*((PI_hatx(2,1))^n10x)*(PI_hatx(2,2)^n11x); 

LR_CCX = -2*log(L_px / L_pix);


if LR_CCX > x95_2
    fprintf('LR_CC GARCH-X test: Reject H0 of correct conditional coverage at 5%% level.\n');
else
    fprintf('LR_CC GARCH-X test: Fail to reject H0 of correct conditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_CC = %.4f, critical value = %.4f (chi^2, df=2)\n\n', ...
        LR_CCX, x95_2);

% unconditional coverage
pi_hatx = n1x/(n0x+n1x);
L_pi_hatx = (1-pi_hatx)^n0x*pi_hatx^n1x;
LR_UCX = -2*log(L_px / L_pi_hatx);


if LR_UCX > x95_1
    fprintf('LR_UC GARCH-X test: Reject H0 of correct unconditional coverage at 5%% level.\n');
else
    fprintf('LR_UC GARCH-X test: Fail to reject H0 of correct unconditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_UC = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_UCX, x95_1);

% independence

LR_INDX = LR_CCX-LR_UCX;


if LR_INDX > x95_1
    fprintf('LR_IND GARCH-X test: Reject H0 of independent interval violations at 5%% level.\n');
else
    fprintf('LR_IND GARCH-X test: Fail to reject H0 of independent interval violations at 5%% level.\n');
end
fprintf('  Test statistic LR_IND = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_INDX, x95_1);


% GARCH 
I_garch = zeros(size(r_realized));

for (i=1:T)
    if (r_realized(i) >= CI_low_garch(i)) && (r_realized(i) <= CI_high_garch(i))
    I_garch(i) = 1;
    end
end 


% conditional coverage
p = 0.95; 

n1 = sum(I_garch);         
n0 = T-n1; 

L_p = (1-p)^n0*p^n1;


n00 = 0;
n01 = 0;
n10 = 0;
n11 = 0; 

for(i=1:T-1)
    if (I_garch(i)== 0 && I_garch(i+1)==0)
        n00=n00+1;
    elseif (I_garch(i)== 0 && I_garch(i+1)==1)
        n01=n01+1;
    elseif(I_garch(i)== 1 && I_garch(i+1)==0)
        n10=n10+1;
    else
        n11=n11+1;
    end 
end 

PI_hat = [n00/(n00+n01) n01/(n00+n01);
    n10/(n10+n11)  n11/(n10+n11) ];

L_pi = ((PI_hat(1,1))^n00)*(PI_hat(1,2)^n01)*((PI_hat(2,1))^n10)*(PI_hat(2,2)^n11); 

LR_CCG = -2*log(L_p / L_pi);


if LR_CCG > x95_2
    fprintf('LR_CC GARCH test: Reject H0 of correct conditional coverage at 5%% level.\n');
else
    fprintf('LR_CC GARCH test: Fail to reject H0 of correct conditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_CC = %.4f, critical value = %.4f (chi^2, df=2)\n\n', ...
        LR_CCG, x95_2);

% unconditional coverage
pi_hat = n1/(n0+n1);
L_pi_hat = (1-pi_hat)^n0*pi_hat^n1;
LR_UCG = -2*log(L_p / L_pi_hat);


if LR_UCG > x95_1
    fprintf('LR_UC GARCH test: Reject H0 of correct unconditional coverage at 5%% level.\n');
else
    fprintf('LR_UC GARCH test: Fail to reject H0 of correct unconditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_UC = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_UCG, x95_1);

% independence

LR_INDG = LR_CCG-LR_UCG;


if LR_INDG > x95_1
    fprintf('LR_IND GARCH test: Reject H0 of independent interval violations at 5%% level.\n');
else
    fprintf('LR_IND GARCH test: Fail to reject H0 of independent interval violations at 5%% level.\n');
end
fprintf('  Test statistic LR_IND = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_INDG, x95_1);


% EGARCH 

I_egarch = zeros(size(r_realized));

for (i=1:n-1)
    if(r_realized(i)>=CI_low_eg(i) && r_realized(i) <= CI_high_eg(i))
        I_egarch(i) = 1;
    end 
end 


% conditional coverage
n1e = sum(I_egarch);         
n0e = T-n1e; 

L_pe = (1-p)^n0e*p^n1e;


n00e = 0;
n01e = 0;
n10e = 0;
n11e = 0; 

for(i=1:T-1)
    if (I_egarch(i)== 0 && I_egarch(i+1)==0)
        n00e=n00e+1;
    elseif (I_egarch(i)== 0 && I_egarch(i+1)==1)
        n01e=n01e+1;
    elseif(I_egarch(i)== 1 && I_egarch(i+1)==0)
        n10e=n10e+1;
    else
        n11e=n11e+1;
    end 
end 

PI_hate = [n00e/(n00e+n01e) n01e/(n00e+n01e);
    n10e/(n10e+n11e)  n11e/(n10e+n11e) ];

L_pie = ((PI_hate(1,1))^n00e)*(PI_hate(1,2)^n01e)*((PI_hate(2,1))^n10e)*(PI_hate(2,2)^n11e); 

LR_CCGe = -2*log(L_pe / L_pie);


if LR_CCGe > x95_2
    fprintf('LR_CC EGARCH test: Reject H0 of correct conditional coverage at 5%% level.\n');
else
    fprintf('LR_CC EGARCH test: Fail to reject H0 of correct conditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_CC = %.4f, critical value = %.4f (chi^2, df=2)\n\n', ...
        LR_CCGe, x95_2);

% unconditional coverage
pi_hate = n1e/(n0e+n1e);
L_pi_hate = (1-pi_hate)^n0*pi_hate^n1e;
LR_UCGe = -2*log(L_pe / L_pi_hate);


if LR_UCGe > x95_1
    fprintf('LR_UC EGARCH test: Reject H0 of correct unconditional coverage at 5%% level.\n');
else
    fprintf('LR_UC EGARCH test: Fail to reject H0 of correct unconditional coverage at 5%% level.\n');
end
fprintf('  Test statistic LR_UC = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_UCGe, x95_1);

% independence

LR_INDGe = LR_CCGe-LR_UCGe;


if LR_INDGe > x95_1
    fprintf('LR_IND EGARCH test: Reject H0 of independent interval violations at 5%% level.\n');
else
    fprintf('LR_IND EGARCH test: Fail to reject H0 of independent interval violations at 5%% level.\n');
end
fprintf('  Test statistic LR_IND = %.4f, critical value = %.4f (chi^2, df=1)\n\n', ...
        LR_INDGe, x95_1);



%% Functions

function [xout,lOut,CovM] = estimation2(initial,data)
    [xout,lOut,CovM] = MLmax(@garchXLL,initial,data);
end

function LL = garchXLL(theta, X) 

    x = X(:,1);
    ext = X(:,2); 
    omega = theta(1);
    alpha = theta(2);
    beta = theta(3);
    gamma = theta(4);
    mu = theta(5);

    T = length(x);
    eps = x - mu;
    if omega <= 0 || alpha < 0 || beta < 0  
        LL = -1e40 * ones(T,1);   
        return;
    end
    sigma2 = zeros(T,1);
    EX = mean(ext);
    sigma2(1) = (omega + gamma*EX) / (1 - alpha - beta); 
    for t = 2:T
        sigma2(t) = omega + alpha*eps(t-1).^2 + beta*sigma2(t-1) + gamma*ext(t-1);
        if sigma2(t) <= 0
            LL = -1e40 * ones(T,1);
            return;
        end
    end
    LL = -0.5 * (log(2*pi) + log(sigma2) + eps.^2 ./ sigma2 ); % assuming normal distribution
end


function LL = garchLL(theta, x)
    x = x(:);
    mu = theta(1);
    omega = theta(2);
    alpha = theta(3);
    beta = theta(4);

    T = length(x);
    eps = x - mu;

    if omega <= 0 || alpha < 0 || beta < 0 

        LL = -1e30 * ones(T,1);   
        return;
    end

    sigma2 = zeros(T,1);
    sigma2(1) = omega/(1-alpha-beta); 

    for t = 2:T
        sigma2(t) = omega + alpha*eps(t-1).^2 + beta*sigma2(t-1);
        if sigma2(t) <= 0
            LL = -1e30 * ones(T,1);
            return;
        end
    end

    LL = -0.5 * (log(2*pi) + log(sigma2) + eps.^2 ./ sigma2 );
end


function LL = EGARCHLL(theta, r)
    r = r(:);

    mu = theta(1);
    omega = theta(2);
    alpha = theta(3);
    beta = theta(4);

    T   = length(r);
    eps = r - mu;

    if abs(beta) >= 0.9999 
        LL = -1e30 * ones(T,1);
        return;
    end

    h = zeros(T,1);

    h(1) = log(var(eps));

    sigma2 = exp(h(1));
    if ~isfinite(sigma2) || sigma2 <= 0
        LL = -1e30 * ones(T,1);
        return;
    end
    w_prev = eps(1) / sqrt(sigma2);  

    for t = 2:T
        h(t) = omega + alpha*w_prev + beta*h(t-1);

        sigma2 = exp(h(t));
        if ~isfinite(sigma2) || sigma2 <= 0
            LL = -1e30 * ones(T,1);
            return;
        end

        w_prev = eps(t) / sqrt(sigma2);  
    end

    sigma2 = exp(h);
    LL = -0.5 * (log(2*pi) + log(sigma2) + (eps.^2) ./ sigma2);

end

