%% Financial Statistics Project Part 3
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

addpath('/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/')

%% 3: Portfolio Optimization, Track A
clear;
ASSETS=readtable('ASSETSA.csv'); 

no = 12;

Date=ASSETS.Date; %2015-01-28 to 2020-01-28  
PRICES=table2array(ASSETS(:,2:13));
LOGRETURNS=diff(log(PRICES));

FULLNAMES={'The Allstate Corporation (ALL)',...
'Vanguard Total Bond Market Index Fund (BND)',...
'Bitcoin USD Price (BTC-USD)',...
'Intel Corporation (INTC)',...
'Coffee futures (KC=F)',...
'NextEra Energy, Inc. (NEE)',...
'NVIDIA Corporation (NVDA)',...
'Silver futures (SI=F)',...
'Sysco Corporation (SYY)',...
'UnitedHealth Group Incorporated (UNH)',...
'Walmart Inc. (WMT)',...
'Exxon Mobil Corporation (XOM)'};
figure(1)
for i=1:no
subplot(4,3,i)
plot(Date,PRICES(:,i))
title(FULLNAMES{i})
end
figure(2)
for i=1:no
subplot(4,3,i)
plot(Date(2:end),LOGRETURNS(:,i))
title(FULLNAMES{i})
end

N = length(LOGRETURNS);
N_training = 738; 

%% Start-up

assetprices = ASSETS{:,2:end};
assetprices(462,5)= 154.65; % we looked up the value because it is nan
r_k = diff(log(assetprices)); 
r_k_train = r_k(1:N_training,:);
r_k_test = r_k(N_training + 1: end,:);

mu = mean(r_k_train); % unconditional average return of each asset 
sigma = std(r_k_train); % unconditional volatility of each asset

r_k_train_dm = r_k_train - mean(r_k_train,1);

covariance = cov(r_k_train); 

correlation = corr(r_k_train);

figure;

h = heatmap(FULLNAMES, FULLNAMES, correlation);
h.Colormap = cool; 
lim = max(abs(correlation(:)));
h.ColorLimits = [-lim, lim];
h.ColorbarVisible = 'on';
title('Correlation matrix of asset returns');

alpha = 0.05;
rf_yearly = 0.045;    


%% Distributions

figure;
for i=1:no
    subplot(3,4,i)
    assetsdemeaned = (r_k_train(:,i) - mu(:,i)) ./ sigma(:,i);

    nDist = fitdist(assetsdemeaned, 'Normal');
    tDist = fitdist(assetsdemeaned, 'tLocationScale');
    
    Nnll = negloglik(nDist);
    tnll = negloglik(tDist);
    histogram(assetsdemeaned, 'Normalization', 'pdf','FaceAlpha', 0.4,'EdgeColor', 'none');

    hold on;
    x = linspace(min(assetsdemeaned), max(assetsdemeaned), 100);
    nP = plot(x, pdf(nDist, x), 'b', 'LineWidth', 1.5);
    tP = plot(x, pdf(tDist, x), 'r', 'LineWidth', 1.5);
    hold off;
    
    title(sprintf('%s\n Normal: %.2f\n Student-t: %.2f', FULLNAMES{i}, -Nnll, -tnll), 'FontWeight','normal');

    xlabel('Demeaned return');
    ylabel('Density');
end
legend([nP, tP], 'Normal', 'Student-t');
sgtitle('Asset return distributions and log-likelihoods', 'FontWeight', 'bold');

% student-t is better 


%% Risk Aversion Parameter

one = ones(no,1);

sigmainv = covariance \ eye(no);

A = one' * sigmainv * one;        
B = one' * sigmainv * mu';          

R_GMV = B / A;                 
V_GMV = 1 / A;                  

R = sigmainv - (sigmainv*one*one'*sigmainv) / A;
s = mu*R*mu';

d = norminv(1 - alpha); 

Er2 = 1; 

c = (d^2) / Er2;                  
if c <= s
    warning('condition fails: d^2/E[r^2] <= s. gamma not well-defined here.');
end

kappa = sqrt(V_GMV) / (c - s);   
gamma = 1 / ( 1 + R_GMV + (1+s)*kappa );

fprintf('The risk aversion paramter: gamma = %.6g\n', gamma);

%% Alternativ risk aversion parameter approach 

gamma_grid = [1 2 4 6 8 10 15 18 20 25 30 40 50];
nGamma     = length(gamma_grid);
nAssets    = length(mu);

W      = zeros(nAssets, nGamma);
ret    = zeros(nGamma,1);
vol    = zeros(nGamma,1);
sharpe = zeros(nGamma,1);

options = optimoptions('quadprog','Display','off');

Aeq = ones(1,nAssets);
beq = 1;
lb  = zeros(nAssets,1);

for i = 1:nGamma
    gamma = gamma_grid(i);

    H = gamma * covariance;
    f = -mu;

    w = quadprog(H,f,[],[],Aeq,beq,lb,[],[],options);

    W(:,i) = w;

    ret(i) = mu * w;
    vol(i) = sqrt(w' * covariance * w);
    sharpe(i) = ret(i) / vol(i);
end
results = table(gamma_grid',ret,vol,sharpe,...
    'VariableNames',{'Gamma','MeanReturn','Volatility','Sharpe'});

disp(results)
figure;
plot(gamma_grid, W','LineWidth',1.2);
xlabel('\gamma'); ylabel('Portfolio Weight');
title('Portfolio Weights Across Risk Aversion Levels');
grid on;

%% Conditional mean 

p = 1;
figure;

for i=1:12
    subplot(3, 4, i);
    autocorr(r_k_train_dm(:,i).^2);
    title(FULLNAMES{i} + "ACF", 'FontWeight','normal');
    disp(FULLNAMES{i});
    checkWhiteACF(r_k_train_dm(:,i), alpha, p);

end
sgtitle('ACF of Daily Returns');


figure;
for i = 1:12
    subplot(3,4,i);
    parcorr(r_k_train_dm(:,i), 'NumLags', 20);
    title([FULLNAMES{i} ' PACF'], 'FontWeight','normal');
end
sgtitle('PACF of Daily Returns');

%% Model coffee futures

y = r_k_train_dm(:,5);

cands = {
    arima('Constant',NaN,'ARLags',1,'MALags',[])
    arima('Constant',NaN,'ARLags',1:2,'MALags',[])
    arima('Constant',NaN,'ARLags',1:3,'MALags',[])
    arima('Constant',NaN,'ARLags',1,'MALags',1)
};

T = numel(y);

for j = 1:numel(cands)
    [fit,~,LL] = estimate(cands{j}, y, 'Display','off');

    [res,~] = infer(fit,y);
    [~,pval] = lbqtest(res,'Lags',20);
    p = fit.P;
    q = fit.Q;
    k = 1 + p + q + 1;   % constant + AR + MA + variance

    [AIC,BIC] = aicbic(LL, k, T);

    fprintf('%d: AR(%d) MA(%d) | AIC=%.2f  BIC=%.2f  LB-p=%.4f\n', ...
            j, p, q, AIC, BIC, pval);
end

 a = arima('Constant',NaN,'ARLags',1,'MALags',[]);
 est = estimate(a, y, 'Display','off');

coffee_param = est.AR{1};

%% Model Exxon 

y = r_k_train_dm(:,12);     

mdl = arima('Constant', NaN,'ARLags',1:3,'MALags',[]);
est = estimate(mdl, y, 'Display','off');

[res,~] = infer(est, y);

[h,pval] = lbqtest(res, 'Lags', 20);
fprintf('XOM AR(1) Ljung-Box p=%.4f\n', pval);


y = r_k_train(:,12);   
y = y(:);

fits = cell(3,1);
LL   = zeros(3,1);
k    = zeros(3,1);

for p = 1:3
    model = arima('Constant',NaN,'ARLags',1:p,'MALags',[]);
   [fits{p},~,LL(p)]  = estimate(model, y, 'Display','off');
    k(p)  = 1 + p + 1; % constant + AR params + variance
end

T = length(y);
AIC = -2*LL + 2*k;
BIC = -2*LL + log(T)*k;

table((1:3)',AIC,BIC,'VariableNames',{'p','AIC','BIC'})

exxon_params = est.AR;

%% Conditional mean modeling for test data 

t0 = N_training + 1;   
t1 = size(r_k,1);      
Ttest = t1 - t0 + 1;

mu_hat_test = zeros(Ttest, no);

coffeeIdx = 5;
xomIdx    = 12;

for tt = 1:Ttest
    t = t0 + tt - 1;           
    r_hist = r_k(1:t-1, :);   

    mu_t = mean(r_hist, 1);
    
    % Coffee 
    rc = mu_t(coffeeIdx);
    rlagC = r_k(t-1, coffeeIdx);
    mu_t(coffeeIdx) = rc + coffee_param*(rlagC - rc);

    % Exxon
     rx = mu_t(xomIdx);
    rlag1 = r_k(t-1, xomIdx);
    rlag2 = r_k(t-2, xomIdx);
    rlag3 = r_k(t-3, xomIdx);
    phi1 = exxon_params{1};
    phi2 = exxon_params{2};
    phi3 = exxon_params{3};
    mu_t(xomIdx) = rx + phi1*(rlag1 - rx)+ phi2 * (rlag2 - rx) +phi3*(rlag3 - rx);

    mu_hat_test(tt, :) = mu_t;
end


%% Conditional covariance (Normal distribution)

omega0 = 0.1 * var(r_k_train_dm)';   
alpha0 = 0.1 * ones(no,1); 
beta0  = 0.8 * ones(no,1); 

R0 = correlation;
R0 = (R0 + R0')/2;
R0(1:no+1:end) = 1;

theta0 = [omega0; alpha0; beta0; corr_vech(R0)];


[theta_hat, logL, covM] = MLmax(@cccmvgarchLL, theta0, r_k_train_dm); 


[omega_hat, alpha_hat, beta_hat, R_hat] = unpack_ccc_params(theta_hat, no);
disp('Estimated constant correlation matrix R:');
disp(R_hat);

 h = zeros(N_training,no);
 h(1,:) = var(r_k_train_dm);      

 for t = 2:N_training
      h(t,:) = omega_hat' + alpha_hat'.*(r_k_train_dm(t-1,:).^2) + beta_hat'.*h(t-1,:);
 end

figure;
plot(1:N_training, log(sqrt(h)), 'LineWidth', 1.2);
grid on;

xlabel('Time');
ylabel('Log Conditional Volatility');
title('Estimated Conditional Log Volatilities of Training Data');
legend(FULLNAMES);


%% Testing Gaussian CCC-GARCH 


[omega_hat, alpha_hat, beta_hat, R_hat] = unpack_ccc_params(theta_hat, no);


z = r_k_train_dm ./ sqrt(h); % standardized residuals  

R_hat = (R_hat + R_hat')/2;
R_hat(1:no+1:end) = 1;
[Lr,p] = chol(R_hat,'lower');

u = (Lr \ z')';              

kurt_u  = kurtosis(u, 0, 1);
exkurt  = kurt_u - 3;

fprintf('Excess kurtosis summary (u):\n');
fprintf('  min=%.3f, median=%.3f, max=%.3f\n', min(exkurt), median(exkurt), max(exkurt));

y = (Lr \ z')';           
q = sum(y.^2, 2);                 


q_sorted = sort(q);
pgrid    = ((1:numel(q_sorted))' - 0.5) / numel(q_sorted);
q_theory = chi2inv(pgrid, N);

figure;
plot(q_theory, q_sorted, '.'); grid on;
xlabel(sprintf('Theoretical Chi-square(%d) quantiles', N));
ylabel('Empirical q_t quantiles');
title(sprintf('CCC-GARCH Gaussian check: q_t vs Chi-square(%d)', N));

%% CCC-GARCH student-t
% This takes a really long time to do 

omega0 = 0.1 * var(r_k_train)';     
alpha0 = 0.10 * ones(no, 1);
beta0  = 0.80 * ones(no, 1);

R0 = correlation;
R0 = (R0 + R0')/2;
R0(1:no+1:end) = 1;

nu0 = 8;                       
nu0_raw = nu0 - 2;         

theta0 = [omega0; alpha0; beta0; corr_vech(R0); nu0_raw];

[theta_hat, logL, covM] = MLmax(@cccmvgarchLL_t, theta0, r_k_train);

[omega_hat, alpha_hat, beta_hat, R_hat, nu_hat] = unpack_ccc_params_t(theta_hat, no);

fprintf('Estimated df (nu): %.3f\n', nu_hat);
disp('Estimated constant correlation matrix R_hat:');
disp(R_hat);

h = zeros(N_training, no);
h(1,:) = var(r_k_train);

for t = 2:N_training
    h(t,:) = omega_hat(:)' + alpha_hat(:)'.*(r_k_train(t-1,:).^2) + beta_hat(:)'.*h(t-1,:);
end

figure;
plot(1:N_training, sqrt(h), 'LineWidth', 1.2);
grid on;
xlabel('Time');
ylabel('Conditional Volatility');
title('Estimated Conditional Volatilities (Training Sample)');
legend(FULLNAMES, 'Location', 'best');

%% Testing Student-t CCC-GARCH 


z = r_k_train_dm ./ sqrt(h); 

R_hat = (R_hat + R_hat')/2;
R_hat(1:no+1:end) = 1;
[Lr,p] = chol(R_hat,'lower');

y = (Lr \ z')'; 

figure;

for i = 1:no
    subplot(3,4,i);

    yi = sort(y(:,i));
    T  = numel(yi);

    pgrid = ((1:T)' - 0.5) / T;

    q_theory = tinv(pgrid, nu_hat);

    plot(q_theory, yi, '.', 'MarkerSize', 8);
    hold on;

    xlim_auto = xlim;
    plot(xlim_auto, xlim_auto, 'k--', 'LineWidth', 1);
    hold off;

    grid on;
    title(FULLNAMES{i}, 'FontWeight','normal');
    xlabel('Theoretical t quantiles');
    ylabel('Empirical residuals');
end

sgtitle(sprintf('Univariate Q–Q plots vs Student-t (\\nu = %.2f)', nu_hat));

%% Maximizing the unconditional portfolio 

V_0 = 1e6; % start with arbitrary value of 1M 
gamma = 19;

compute_weights = @(mu_hat,Sigma_hat) ...
    quadprog( gamma*(Sigma_hat+Sigma_hat')/2, ...  
              -mu_hat(:), -eye(no), zeros(no,1), ones(1,no), 1, ...                    
              [], [], [], optimoptions('quadprog','Display','off') );

%% Static unconditional

V = zeros(2,1); % portfolio values over the periods
V(1) = V_0;

muvec = mu(:);  

w_0 = compute_weights(muvec, covariance);


pricest = assetprices(N_training,:)';
shares_0  = (V_0 * w_0) ./ pricest;

prices_end = assetprices(end,:)';  
V(2) = shares_0' * prices_end;

profit_test = (V(2)-V(1))/V(1);

fprintf('Portfolio value at %s: %.2f\n', string(Date(N)), V(2));
fprintf('Profit over testing period: %.4f (%.2f%%)\n', profit_test, 100*profit_test);

% Sharpe ratio is not possibe with only one value

%% Yearly unconditional

t0 = N_training;            
t1 = 989;  % after 1 year in testing (calender years for good measure)
t2 = 1241; % after 2 years in testing 
t3 = N; 

V_yearly = zeros(4,1); % portfolio values over the periods
V_yearly(1) = V_0;

P1 = assetprices(t1, :)';
V1 = shares_0' * P1;
V_yearly(2) = V1;

r_up_to_t1 = r_k(1:(t1-1), :);
mu1 = mean(r_up_to_t1)';               
S1  = cov(r_up_to_t1);                 

w1 = compute_weights(mu1, S1);

shares_1 = (V1 * w1) ./ P1;

P2 = assetprices(t2, :)';
V2 = shares_1' * P2;
V_yearly(3) = V2;

r_up_to_t2 = r_k(1:(t2-1), :);    
mu2 = mean(r_up_to_t2)';
S2  = cov(r_up_to_t2);

w2 = compute_weights(mu2, S2);

shares_2 = (V2 * w2) ./ P2;

PN = assetprices(t3, :)';
VN = shares_2' * PN;
V_yearly(4) = VN;

total_return = (VN - V_0) / V_0;

fprintf('Portfolio value at %s: VN = %.2f\n', string(Date(t3)), V_yearly(4));
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

% Sharpe ratio 
r_yearly = (V_yearly(2:end) - V_yearly(1:end-1) )./ V_yearly(1:end-1);
excess = r_yearly - rf_yearly;
sharpe_annual = mean(excess) / std(excess);

fprintf('Annual Sharpe ratio = %.4f\n', sharpe_annual);

%% Monthly unconditional

% there are 25 months in the testing set and assuming equal lengths for
% simplicity

t0 = N_training;   
tend = N;            
M  = 25; % there will be 24 rebalances (none in the last month)          

test_len_prices = tend - t0 + 1;                
step = floor(test_len_prices / M);             

rebalance_idx = t0  + (1:(M-1)) * step;
rebalance_idx = rebalance_idx(rebalance_idx < tend);


V_monthly = zeros(numel(rebalance_idx) + 2, 1);
V_monthly(1) = V_0;

shares_curr = zeros(no,numel(rebalance_idx) + 1);
shares_curr(:,1) = shares_0(:);

weights = zeros(no,numel(rebalance_idx) + 1);
weights(:,1) = w_0;

k = 1;
for t = rebalance_idx(:)'

    Pt = assetprices(t,:)';
    Vt = shares_curr(:,k)' * Pt;
    V_monthly(1 + k) = Vt;

    r_up_to_t = r_k(1:(t-1), :);
    mu_t = mean(r_up_to_t)';
    S_t  = cov(r_up_to_t);

    w_t = compute_weights(mu_t, S_t);
    weights(:,k+1) = w_t;
    shares_curr(:,k+1) = (Vt * w_t) ./ Pt;

    k = k + 1;
end

P_end = assetprices(tend,:)';
V_end = shares_curr(:,end)' * P_end;
V_monthly(end) = V_end;

total_return = (V_end - V_0) / V_0;

fprintf('Final portfolio value at %s: %.2f\n', string(Date(tend)), V_end);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);


dates_port = [Date(t0); Date(rebalance_idx(:)); Date(tend)];

figure;
plot(dates_port, V_monthly, '-o', 'LineWidth', 1.5);
grid on;

xlabel('Date');
ylabel('Portfolio Value');
title('Portfolio Value Over Time (Monthly Rebalancing)');

figure;
plot(dates_port(1:end-1), weights, '-o', 'LineWidth', 1.5);
grid on;
legend(FULLNAMES)

xlabel('Date');
ylabel('Weights of assets');
title('Weights of Assets Over Time (Monthly Rebalancing)');

% Sharpe ratio
yrs = year(dates_port(2:end));
uY = unique(yrs, 'stable');
V_start = zeros(numel(uY),1);
V_end   = zeros(numel(uY),1);
for k = 1:numel(uY)
    idx = find(yrs == uY(k));
    V_start(k) = V_monthly(idx(1));
    V_end(k)   = V_monthly(idx(end));
end
r_annual = (V_end - V_start) ./ V_start;
excess = r_annual - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe (from monthly rebalancing values) = %.4f\n', sharpe_annual);

% VaR
r_monthly = diff(V_monthly) ./ V_monthly(1:end-1);
VaR95_monthly = -quantile(r_monthly, alpha);

fprintf('Monthly VaR 95%% = %.4f\n', VaR95_monthly);


%% Changing to Student-t values

S = load("/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/omega_hat.mat");
omega_hat = S.omega_hat;

S = load("/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/alpha_hat.mat");
alpha_hat = S.alpha_hat;

S = load("/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/beta_hat.mat");
beta_hat = S.beta_hat;

%% Estimate for testing data with CCC-GARCH (unconditional mean) yearly reallocation

x_test   = r_k_test  - mu;

T_test = size(x_test,1);

h_test = zeros(T_test, no);

h0 = omega_hat(:)' ./ max(1 - alpha_hat(:)' - beta_hat(:)', 1e-6);
h_test(1,:) = h0;

for t = 2:T_test
    h_test(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x_test(t-1,:).^2) + beta_hat(:)'.*h_test(t-1,:);
end

t1_full = 989;
t2_full = 1241;
t3_full = N;

t1 = t1_full - N_training;   
t2 = t2_full - N_training;
t3 = t3_full - N_training;

V_yearly = zeros(4,1);
V_yearly(1) = V_0;

P1 = assetprices(t1_full, :)';
V1 = shares_0' * P1;
V_yearly(2) = V1;

sig1 = sqrt(h_test(t1+1,:))';
H1   = diag(sig1) * R_hat * diag(sig1);

mu_1 = mean(r_k(1:t1_full-1, :), 1)';  

w1 = compute_weights(mu_1, H1);    
shares_1 = (V1 * w1) ./ P1;

P2 = assetprices(t2_full, :)';
V2 = shares_1' * P2;
V_yearly(3) = V2;

sig2 = sqrt(h_test(t2+1,:))';
H2   = diag(sig2) * R_hat * diag(sig2);

mu_2 = mean(r_k(1:t2_full-1, :), 1)';  

w2 = compute_weights(mu_2, H2);
shares_2 = (V2 * w2) ./ P2;

PN = assetprices(t3_full, :)';
VN = shares_2' * PN;
V_yearly(4) = VN;

total_return = (VN - V_0) / V_0;

fprintf('Portfolio value at %s: VN = %.2f\n', string(Date(t3_full)), VN);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

% Sharpe ratio 
r_yearly = (V_yearly(2:end) - V_yearly(1:end-1) )./ V_yearly(1:end-1);
excess = r_yearly - rf_yearly;
sharpe_annual = mean(excess) / std(excess);

fprintf('Annual Sharpe ratio = %.4f\n', sharpe_annual);


%% Estimate for testing data with CCC-GARCH (unconditional mean) monthly reallocation

h_test = zeros(T_test, no);

den = max(1 - alpha_hat(:)' - beta_hat(:)', 1e-6);
h0  = omega_hat(:)' ./ den;
h_test(1,:) = h0;

for t = 2:T_test
    h_test(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x_test(t-1,:).^2) + beta_hat(:)'.*h_test(t-1,:);
end

t0   = N_training;      
tend = N;               
M    = 25;             

test_len_prices = tend - t0 + 1;
step = floor(test_len_prices / M);

rebalance_idx_full = t0 + (1:(M-1)) * step;           
rebalance_idx_full = rebalance_idx_full(rebalance_idx_full < tend);

rebalance_idx_test = rebalance_idx_full - N_training;

V_monthly = zeros(numel(rebalance_idx_full) + 2, 1);
dates_port = [Date(t0); Date(rebalance_idx_full(:)); Date(tend)];

V_monthly(1) = V_0;

shares_curr = zeros(no, numel(rebalance_idx_full) + 1);
shares_curr(:,1) = shares_0(:);
weights = zeros(no, numel(rebalance_idx_full) + 1);
weighs(:,1) = w_0;

k = 1;

for j = 1:numel(rebalance_idx_full)
    t_full = rebalance_idx_full(j);
    t_test = rebalance_idx_test(j);

    Pt = assetprices(t_full, :)';
    Vt = shares_curr(:,k)' * Pt;
    V_monthly(1 + k) = Vt;

    mu_t = mean(r_k(1:(t_full-1), :), 1)';       

    sig_t = sqrt(h_test(t_test+1, :))';              
    H_t   = diag(sig_t) * R_hat * diag(sig_t);    

    w_t = compute_weights(mu_t, H_t);
    weights(:,k+1) = w_t; 
    shares_curr(:,k+1) = (Vt * w_t) ./ Pt;

    k = k + 1;
end

P_end = assetprices(tend,:)';
V_end = shares_curr(:,end)' * P_end;
V_monthly(end) = V_end;

total_return = (V_end - V_0) / V_0;

fprintf('Final portfolio value at %s: %.2f\n', string(Date(tend)), V_end);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

figure;
plot(dates_port, V_monthly, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Date');
ylabel('Portfolio Value');
title('Portfolio Value Over Time (Monthly rebalancing, unconditional mean + CCC-GARCH cov)');

figure;
plot(dates_port(1:end-1), weights', 'LineWidth', 1.2);
grid on;
xlabel('Date');
ylabel('Portfolio weights');
title('Portfolio weights (Monthly rebalancing)');
legend(FULLNAMES, 'Location', 'bestoutside');

% Sharpe ratio
yrs = year(dates_port(2:end));
uY = unique(yrs, 'stable');
V_start = zeros(numel(uY),1);
V_end   = zeros(numel(uY),1);
for k = 1:numel(uY)
    idx = find(yrs == uY(k));
    V_start(k) = V_monthly(idx(1));
    V_end(k)   = V_monthly(idx(end));
end
r_annual = (V_end - V_start) ./ V_start;
excess = r_annual - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe (from monthly rebalancing values) = %.4f\n', sharpe_annual);

% VaR
r_monthly = diff(V_monthly) ./ V_monthly(1:end-1);
VaR95_monthly = -quantile(r_monthly, alpha);

fprintf('Monthly VaR 95%% = %.4f\n', VaR95_monthly);


%% Estimate for testing data with CCC-GARCH & conditional mean - yearly reallocation

x_test = r_k_test  - mu;

T_test = size(x_test,1);

h_test = zeros(T_test, no);

h0 = omega_hat(:)' ./ max(1 - alpha_hat(:)' - beta_hat(:)', 1e-6);
h_test(1,:) = h0;

for t = 2:T_test
    h_test(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x_test(t-1,:).^2) + beta_hat(:)'.*h_test(t-1,:);
end

t1_full = 989;
t2_full = 1241;
t3_full = N;

t1 = t1_full - N_training;   
t2 = t2_full - N_training;
t3 = t3_full - N_training;

V_yearly = zeros(4,1);
V_yearly(1) = V_0;

P1 = assetprices(t1_full, :)';
V1 = shares_0' * P1;
V_yearly(2) = V1;

sig1 = sqrt(h_test(t1+1,:))';
H1   = diag(sig1) * R_hat * diag(sig1);

mu_1 = mu_hat_test(t1+1, :)'; 

w1 = compute_weights(mu_1, H1);    
shares_1 = (V1 * w1) ./ P1;

P2 = assetprices(t2_full, :)';
V2 = shares_1' * P2;
V_yearly(3) = V2;

sig2 = sqrt(h_test(t2+1,:))';
H2   = diag(sig2) * R_hat * diag(sig2);

mu_2 = mu_hat_test(t2+1, :)'; 

w2 = compute_weights(mu_2, H2);
shares_2 = (V2 * w2) ./ P2;

PN = assetprices(t3_full, :)';
VN = shares_2' * PN;
V_yearly(4) = VN;

total_return = (VN - V_0) / V_0;

fprintf('Portfolio value at %s: VN = %.2f\n', string(Date(t3_full)), VN);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

% Sharpe ratio 
r_yearly = (V_yearly(2:end) - V_yearly(1:end-1) )./ V_yearly(1:end-1);
excess = r_yearly - rf_yearly;
sharpe_annual = mean(excess) / std(excess);

fprintf('Annual Sharpe ratio = %.4f\n', sharpe_annual);

%% Estimate for testing data with CCC-GARCH & conditional mean monthly reallocation

h_test = zeros(T_test, no);

den = max(1 - alpha_hat(:)' - beta_hat(:)', 1e-6);
h0  = omega_hat(:)' ./ den;
h_test(1,:) = h0;

for t = 2:T_test
    h_test(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x_test(t-1,:).^2) + beta_hat(:)'.*h_test(t-1,:);
end

t0   = N_training;      
tend = N;               
M    = 25;             

test_len_prices = tend - t0 + 1;
step = floor(test_len_prices / M);

rebalance_idx_full = t0 + (1:(M-1)) * step;           
rebalance_idx_full = rebalance_idx_full(rebalance_idx_full < tend);

rebalance_idx_test = rebalance_idx_full - N_training;

V_monthly = zeros(numel(rebalance_idx_full) + 2, 1);
dates_port = [Date(t0); Date(rebalance_idx_full(:)); Date(tend)];

V_monthly(1) = V_0;

shares_curr = zeros(no, numel(rebalance_idx_full) + 1);
shares_curr(:,1) = shares_0(:);
weights = zeros(no, numel(rebalance_idx_full) + 1);
weighs(:,1) = w_0;

k = 1;

for j = 1:numel(rebalance_idx_full)
    t_full = rebalance_idx_full(j);
    t_test = rebalance_idx_test(j);

    Pt = assetprices(t_full, :)';
    Vt = shares_curr(:,k)' * Pt;
    V_monthly(1 + k) = Vt;

    mu_t = mu_hat_test(t_test+1, :)'; 

    sig_t = sqrt(h_test(t_test+1, :))';              
    H_t   = diag(sig_t) * R_hat * diag(sig_t);    

    w_t = compute_weights(mu_t, H_t);
    weights(:,k+1) = w_t; 
    shares_curr(:,k+1) = (Vt * w_t) ./ Pt;

    k = k + 1;
end

P_end = assetprices(tend,:)';
V_end = shares_curr(:,end)' * P_end;
V_monthly(end) = V_end;

total_return = (V_end - V_0) / V_0;

fprintf('Final portfolio value at %s: %.2f\n', string(Date(tend)), V_end);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

figure;
plot(dates_port, V_monthly, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Date');
ylabel('Portfolio Value');
title('Portfolio Value Over Time (Monthly rebalancing, unconditional mean + CCC-GARCH cov)');

figure;
plot(dates_port(1:end-1), weights', 'LineWidth', 1.2);
grid on;
xlabel('Date');
ylabel('Portfolio weights');
title('Portfolio weights (Monthly rebalancing)');
legend(FULLNAMES, 'Location', 'bestoutside');

% Sharpe ratio
yrs = year(dates_port(2:end));
uY = unique(yrs, 'stable');
V_start = zeros(numel(uY),1);
V_end   = zeros(numel(uY),1);
for k = 1:numel(uY)
    idx = find(yrs == uY(k));
    V_start(k) = V_monthly(idx(1));
    V_end(k)   = V_monthly(idx(end));
end
r_annual = (V_end - V_start) ./ V_start;
excess = r_annual - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe (from monthly rebalancing values) = %.4f\n', sharpe_annual);

% VaR
rf_monthly = (1 + rf_yearly)^(1/12) - 1;
r_monthly = diff(V_monthly) ./ V_monthly(1:end-1);
VaR95_monthly = -quantile(r_monthly, alpha);

fprintf('Monthly VaR 95%% = %.4f\n', VaR95_monthly);

%% Rolling window 1 time step update

muvec = mu(:); 
A = -eye(no);    b = zeros(no,1); 
Aeq = ones(1,no); beq = 1;
opts = optimoptions('quadprog','Display','off');
V_0 = 1e6;
V_TC = zeros(length(r_k)-N_training-1);
shares_TC = zeros(length(r_k)-N_training-1,12);
V0_TC = V_0; 
for a = 1:length(r_k)-N_training-1
    muvec = mean(r_k(a:N_training+(a-1),:))';
    covariance = cov(r_k(a:N_training+(a-1),:));
    f = -muvec;
    H = gamma * (covariance + covariance')/2; 
    w_rolling(a,:) = quadprog(H, f, A, b, Aeq, beq, [], [], [], opts);
    prices0_rolling(a,:) = assetprices(N_training+(a-1),:);
    shares_rolling(a,:) = (V_0 * w_rolling(a,:)) ./ prices0_rolling(a,:);
    pricest_rolling(a,:) = assetprices(N_training+a,:);
    V(a,:) = shares_rolling(a,:) * pricest_rolling(a,:)';
    if a == 1
        shares_prev = zeros(1, size(r_k,2));
    else
        shares_prev = shares_rollingTC(a-1,:);
    end
    shares_rollingTC(a,:) = (V0_TC * w_rolling(a,:)) ./ prices0_rolling(a,:);
    TC = sum(abs(shares_rollingTC(a,:) - shares_prev).* prices0_rolling(a,:)*0.03);
    V0_TC = V0_TC - TC;
    shares_rollingTC(a,:) = (V0_TC * w_rolling(a,:)) ./ prices0_rolling(a,:);
    profit_rolling(a) = (V(a)- V_0) / V_0;
    V_TC(a) = shares_rollingTC(a,:) * pricest_rolling(a,:)';
    V_0 = V(a,:);
    x_vector(a) = N_training+(a-1);
end
figure;
plot(ASSETS{:,1}(x_vector),V)
xlabel('Date')
ylabel('Portfolio value')
figure;
plot(ASSETS{:,1}(x_vector),V_TC)
xlabel('Date')
ylabel('Portfolio value minus transaction costs')
disp((V(end)-V(1))/V(1));

%% Analysis of amount of days per month

years = year(ASSETS{:,1});
months = month(ASSETS{:,1});
ym = years*100 + months;
uniqueYM = unique(ym);
uniqueYears = unique(years);
counts = histcounts(ym, [uniqueYM; max(uniqueYM)+1]);
counts_years = histcounts(years, [uniqueYears; max(uniqueYears)+1]);
% there are 25 months in the testing set and assuming equal lengths for
% simplicity
t0 = N_training;
tend = length(LOGRETURNS);            
M  = 25; % there will be 24 rebalances (none in the last month)   
Y = 3;
test_len_prices = tend - t0 + 1;                
step = floor(test_len_prices / M); 
step_years = floor(test_len_prices/Y);
rebalance_idx = t0  + (1:(M-1)) * step;
rebalance_idx = rebalance_idx(rebalance_idx < tend);
rebalance_idx_years = t0  + (1:(Y-1)) * step_years;
rebalance_idx_years = rebalance_idx_years(rebalance_idx_years < tend);


%% Rolling window 1 year  update

A = -eye(no);    b = zeros(no,1);   
Aeq = ones(1,no); beq = 1;
opts = optimoptions('quadprog','Display','off'); 
V_0 = 1e6;

shares_TC_y = zeros(length(r_k)-N_training-1,12);
V0_TC_y = V_0; 
V_TC_y = zeros(numel(rebalance_idx_years), 1);
V_TC_y(1) = V_0;
for a = 1:numel(rebalance_idx_years)
    muvec = mean(r_k(a + (a-1)*251:N_training+(a-1)*251,:))';
    covariance = cov(r_k(a + (a-1)*251:N_training+(a-1)*251,:));
    f = -muvec;
    H = gamma * (covariance + covariance')/2; 
    w_rolling_y(a,:) = quadprog(H, f, A, b, Aeq, beq, [], [], [], opts);
    prices0_rolling_y(a,:) = assetprices(N_training+(a-1)*251,:);
    shares_rolling_y(a,:) = (V_0 * w_rolling_y(a,:)) ./ prices0_rolling_y(a,:);
    pricest_rolling_y(a,:) = assetprices(N_training+a*251,:);
    V_y(a,:) = shares_rolling_y(a,:) * pricest_rolling_y(a,:)';
    if a == 1
        shares_prev = zeros(1, size(r_k,2));
        x_vector_y(a) = N_training;
    else
        shares_prev = shares_rollingTC_y(a-1,:);
        x_vector_y(a) = x_vector(a-1) + 251;
    end
    shares_rollingTC_y(a,:) = (V0_TC_y * w_rolling_y(a,:)) ./ prices0_rolling_y(a,:);
    TC = sum(abs(shares_rollingTC_y(a,:) - shares_prev).* prices0_rolling_y(a,:)*0.03);
    V0_TC_y = V0_TC_y - TC;
    shares_rollingTC_y(a,:) = (V0_TC_y * w_rolling_y(a,:)) ./ prices0_rolling_y(a,:);
    profit_rolling_y(a) = (V_y(a)- V_0) / V_0;
    V_TC_y(a) = shares_rollingTC_y(a,:) * pricest_rolling_y(a,:)';
    V0_TC_y = V_TC_y(a);
    V_0 = V_y(a);
end
pricest_rolling_y(3,:) = assetprices(end,:);
V_y(3) = shares_rolling_y(2,:) * pricest_rolling_y(3,:)';
x_vector_y(3) = length(assetprices);
V_TC_y(3) = shares_rollingTC_y(2,:) * pricest_rolling_y(3,:)';

figure;
plot(ASSETS{:,1}(x_vector_y),V_y,'-o')
grid on
xlabel('Date')
ylabel('Portfolio value')
title('Portfolio vaue over time (Annually rebalancing, rolling window)')
figure;
plot(ASSETS{:,1}(x_vector_y),V_TC_y,'-o')
grid on
xlabel('Date')
ylabel('Portfolio value minus transaction costs')
title('Portfolio vaue over time (Annually rebalancing, rolling window)')
w_rolling_y(3,:) = w_rolling_y(2,:);
figure;
plot(ASSETS{:,1}(x_vector_y), w_rolling_y, 'LineWidth', 1);
xlabel('Date');
ylabel('Portfolio Weight');
title('Asset Weights Over Time');
legend(FULLNAMES, 'Location', 'best');
grid on;
disp((V_y(end)-1e6)/1e6);

% Sharpe ratio 
V_yearly = [V_0; V_y];
r_yearly = (V_yearly(2:end) - V_yearly(1:end-1) )./ V_yearly(1:end-1);
excess = r_yearly - rf_yearly;
sharpe_annual = mean(excess) / std(excess);

fprintf('Annual Sharpe ratio = %.4f\n', sharpe_annual);

%% Rolling window 1 month update
gamma = 19;
A = -eye(no);    b = zeros(no,1); 
Aeq = ones(1,no); beq = 1;
opts = optimoptions('quadprog','Display','off');
V_0 = 1e6;
V_TC_m = zeros(numel(rebalance_idx),1);
shares_TC_m = zeros(length(r_k)-N_training-1,12);
V0_TC_m = V_0; 
V_monthlyTC_m = zeros(numel(rebalance_idx), 1);
V_monthlyTC_m(1) = V_0;
for a = 1:numel(rebalance_idx)
    muvec = mean(r_k(a + (a-1)*20:N_training+(a-1)*20,:))';
    covariance = cov(r_k(a + (a-1)*20:N_training+(a-1)*20,:));
    f = -muvec;
    H = gamma * (covariance + covariance')/2; 
    w_rolling_m(a,:) = quadprog(H, f, A, b, Aeq, beq, [], [], [], opts);
    prices0_rolling_m(a,:) = assetprices(N_training+(a-1)*20+1,:);
    shares_rolling_m(a,:) = (V_0 * w_rolling_m(a,:)) ./ prices0_rolling_m(a,:);
    pricest_rolling_m(a,:) = assetprices(N_training+a*20,:);
    V_m(a,:) = shares_rolling_m(a,:) * pricest_rolling_m(a,:)';
    if a == 1
        shares_prev = zeros(1, size(r_k,2));
        x_vector_m(a) = N_training;
    else
        shares_prev = shares_rollingTC_m(a-1,:);
        x_vector_m(a) = x_vector_m(a-1) + 20;
    end
    shares_rollingTC_m(a,:) = (V0_TC_m * w_rolling_m(a,:)) ./ prices0_rolling_m(a,:);
    TC = sum(abs(shares_rollingTC_m(a,:) - shares_prev).* prices0_rolling_m(a,:)*0.03);
    V0_TC_m = V0_TC_m - TC;
    shares_rollingTC_m(a,:) = (V0_TC_m * w_rolling_m(a,:)) ./ prices0_rolling_m(a,:);
    profit_rolling_m(a) = (V(a)- V_0) / V_0;
    V_TC_m(a,:) = shares_rollingTC_m(a,:) * pricest_rolling_m(a,:)';
    V_0 = V_m(a,:);
end
pricest_rolling_m(25,:) = assetprices(end,:);
V_m(25) = shares_rolling_m(24,:) * pricest_rolling_m(25,:)';
x_vector_m(25) = length(assetprices);
V_TC_m(25) = shares_rollingTC_m(24,:) * pricest_rolling_m(25,:)';
figure;
plot(ASSETS{:,1}(x_vector_m),V_m,'-o')
grid on
xlabel('Date')
ylabel('Portfolio value')
title('Portfolio vaue over time (Monthly rebalancing, rolling window)')
figure;
plot(ASSETS{:,1}(x_vector_m),V_TC_m,'-o')
grid on
xlabel('Date')
ylabel('Portfolio value minus transaction costs')
title('Portfolio vaue over time (Monthly rebalancing, rolling window)')
w_rolling_m(25,:) = w_rolling_m(24,:);
figure;
plot(ASSETS{:,1}(x_vector_m), w_rolling_m, 'LineWidth', 1.5);
xlabel('Date');
ylabel('Portfolio Weight');
title('Asset Weights Over Time');
legend(FULLNAMES, 'Location', 'best');
grid on;
disp((V_m(end)-1e6)/1e6);

% Sharpe ratio
yrs = year(dates_port(2:end));
uY = unique(yrs, 'stable');
V_start = zeros(numel(uY),1);
V_end   = zeros(numel(uY),1);
V_monthly = [1000000; V_m];
for k = 1:numel(uY)
    idx = find(yrs == uY(k));
    V_start(k) = V_monthly(idx(1));
    V_end(k)   = V_monthly(idx(end));
end
r_annual = (V_end - V_start) ./ V_start;
excess = r_annual - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe (from monthly rebalancing values) = %.4f\n', sharpe_annual);

% VaR
r_monthly = diff(V_monthly) ./ V_monthly(1:end-1);
VaR95_monthly = -quantile(r_monthly, alpha);

fprintf('Monthly VaR 95%% = %.4f\n', VaR95_monthly);



%% DCC-GARCH(1,1) (Conditional covariance)

V_0 = 1000000;

x = r_k_train;                             
x = x - mean(x,1);                          

omega0 = 0.1*var(x)';                 
w0 = log(omega0);

alpha0 = 0.05*ones(no,1);
beta0  = 0.90*ones(no,1);
alpha_tilde0 = log(alpha0*10);
beta_tilde0 = log(beta0*10); % transform for parameter restrictions

a0 = 0.01; 
b0 = 0.98;
gamma_tilde0 = log(a0*10);
delta_tilde0 = log(b0*10);

theta0 = [w0; alpha0; beta0; gamma_tilde0; delta_tilde0];

theta_hat = MLmax(@dcc_garch_ll, theta0, x);

[omega_hat, alpha_hat, beta_hat, a_hat, b_hat] = unpack_dcc_garch(theta_hat, no);
disp("Mean alpha, beta:"); 
disp([mean(alpha_hat), mean(beta_hat)]);
disp(["a=", a_hat, "b=", b_hat, "a+b=", a_hat+b_hat]);


%% DCC monthly with normal distribution 

h_tr = zeros(N_training, no);

den_tr = max(1 - alpha_hat(:)' - beta_hat(:)', 1e-6);
h_tr(1,:) = omega_hat(:)' ./ den_tr;

for t=2:N_training
    h_tr(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x(t-1,:).^2) + beta_hat(:)'.*h_tr(t-1,:);
end

eps_tr = x ./ sqrt(h_tr);           
Qbar   = (eps_tr'*eps_tr)/N_training;    
Qbar   = (Qbar + Qbar')/2;

Qt_last  = Qbar;                       
eps_last = eps_tr(end,:)';              
h_last   = h_tr(end,:)';                


x_te = r_k_test - mean(r_k_train,1);   
Tte  = size(x_te,1);

h_test  = zeros(Tte, no);
eps_te  = zeros(Tte, no);
Rt_test = zeros(no, no, Tte);
H_test  = zeros(no, no, Tte);

h_test(1,:) = h_last(:)';

Qt = Qt_last;

for t=1:Tte
    if t>=2
        h_test(t,:) = omega_hat(:)' + alpha_hat(:)'.*(x_te(t-1,:).^2) + beta_hat(:)'.*h_test(t-1,:);
    end
    h_test(t,:) = max(h_test(t,:), 1e-12);

    eps_te(t,:) = x_te(t,:) ./ sqrt(h_test(t,:));

    if t==1
        eprev = eps_last;               
    else
        eprev = eps_te(t-1,:)';
    end
    Qt = (1-a_hat-b_hat)*Qbar + a_hat*(eprev*eprev') + b_hat*Qt;
    Qt = (Qt + Qt')/2;

    dq = sqrt(max(diag(Qt), 1e-12));
    Dq_inv = diag(1./dq);
    Rt = Dq_inv * Qt * Dq_inv;
    Rt = (Rt + Rt')/2;
    Rt(1:no+1:end) = 1;

    [L,p] = chol(Rt,'lower');
    if p>0
      error('Rt not PD on test at t=%d. Try smaller a_hat+b_hat or larger jitter.', t);
    end

    Rt_test(:,:,t) = Rt;

    sig = sqrt(h_test(t,:))';
    H_test(:,:,t) = diag(sig) * Rt * diag(sig);
end


t0   = N_training;
tend = size(assetprices,1);
M    = 25;

test_len_prices = tend - t0 + 1;
step = floor(test_len_prices / M);

rebalance_idx_full = t0 + (1:(M-1))*step;
rebalance_idx_full = rebalance_idx_full(rebalance_idx_full < tend);

rebalance_idx_test = rebalance_idx_full - N_training;  

V_monthly = zeros(numel(rebalance_idx_full)+2,1);
dates_port = [Date(t0); Date(rebalance_idx_full(:)); Date(tend)];

V_monthly(1) = V_0;

shares_curr = zeros(no, numel(rebalance_idx_full)+1);
shares_curr(:,1) = shares_0(:);

weights = zeros(no, numel(rebalance_idx_full)+1);
weights(:,1) = w_0;

k = 1;

for j = 1:numel(rebalance_idx_full)
    t_full = rebalance_idx_full(j);
    t_test = rebalance_idx_test(j);

    Pt = assetprices(t_full,:)';
    Vt = shares_curr(:,k)' * Pt;
    V_monthly(1+k) = Vt;

    mu_t = mu_hat_test(t_test+1,:)'; % conditional mean
    %mu_t = mean(r_k(1:(t_full-1), :), 1)';  % unconditional mean     

    H_t = H_test(:,:,t_test+1);

    w_t = compute_weights(mu_t, H_t);
    weights(:,k+1) = w_t;
    shares_curr(:,k+1) = (Vt * w_t) ./ Pt;

    k = k + 1;
end

P_end = assetprices(tend,:)';
V_end = shares_curr(:,end)' * P_end;
V_monthly(end) = V_end;

total_return = (V_end - V_0)/V_0;
fprintf('Final portfolio value at %s: %.2f\n', string(Date(tend)), V_end);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

% VaR
r_monthly = diff(V_monthly) ./ V_monthly(1:end-1);
VaR95_monthly = -quantile(r_monthly, alpha);
fprintf('Monthly VaR 95%% = %.4f\n', VaR95_monthly);

yrs = year(dates_port(2:end));
uY = unique(yrs,'stable');

V_start = zeros(numel(uY),1);
V_end_y = zeros(numel(uY),1);

for kk=1:numel(uY)
    idx = find(yrs==uY(kk));
    V_start(kk) = V_monthly(idx(1));
    V_end_y(kk) = V_monthly(idx(end));
end

r_annual = (V_end_y - V_start) ./ V_start;
excess = r_annual - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe (from monthly DCC values) = %.4f\n', sharpe_annual);

figure;
plot(dates_port, V_monthly, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Date');
ylabel('Portfolio Value');
title('Portfolio Value Over Time (Monthly rebalancing, DCC-GARCH Normal)');

dates_w = dates_port(1:end-1);

figure;
plot(dates_w, weights', 'LineWidth', 1.2);
grid on;
xlabel('Date');
ylabel('Portfolio weights');
title('Portfolio Weights Over Time (Monthly rebalancing, DCC-GARCH Normal)');
legend(FULLNAMES, 'Location', 'bestoutside');


%% DCC yearly with normal distribution 

t1_full = 989;
t2_full = 1241;
t3_full = N;          

t1 = t1_full - N_training;   
t2 = t2_full - N_training;
t3 = t3_full - N_training; 

V_yearly = zeros(4,1);
V_yearly(1) = V_0;

P1 = assetprices(t1_full,:)';
V1 = shares_0' * P1;
V_yearly(2) = V1;

mu_1 = mu_hat_test(t1+1,:)'; % Conditional mean
%mu_1 = mean(r_k(1:(t1_full-1),:), 1)'; % Unconditional mean


H1 = H_test(:,:,t1+1);             
w1 = compute_weights(mu_1, H1);
shares_1 = (V1 * w1) ./ P1;

P2 = assetprices(t2_full,:)';
V2 = shares_1' * P2;
V_yearly(3) = V2;

mu_2 = mu_hat_test(t2+1,:)'; % Conditional mean
%mu_2 = mean(r_k(1:(t2_full-1),:), 1)'; % Unconditional mean

H2 = H_test(:,:,t2+1);
w2 = compute_weights(mu_2, H2);
shares_2 = (V2 * w2) ./ P2;

PN = assetprices(t3_full,:)';
VN = shares_2' * PN;
V_yearly(4) = VN;

total_return = (VN - V_0)/V_0;
fprintf('Portfolio value at %s: VN = %.2f\n', string(Date(t3_full)), VN);
fprintf('Profit over testing period: %.4f (%.2f%%)\n', total_return, 100*total_return);

r_yearly = diff(V_yearly) ./ V_yearly(1:end-1);
excess   = r_yearly - rf_yearly;
sharpe_annual = mean(excess) / std(excess);
fprintf('Annual Sharpe ratio from DCC = %.4f\n', sharpe_annual);



%% Functions 

function LL = cccmvgarchLL(theta, x)
    [T, N] = size(x);

    [omega, alpha, beta, R] = unpack_ccc_params(theta, N);

    if any(omega <= 0) || any(alpha <= 0) || any(beta <= 0) || any(alpha + beta >= 1)
        LL = -1e40 * ones(1,T); 
        return;
    end

    if any(abs(R(:)) > 1.000) || any(abs(diag(R) - 1) > 1e-6)
        LL = -1e40 * ones(1,T); 
        return;
    end
    
    [Lr,p] = chol((R+R')/2, 'lower');
    if p > 0
        LL = -1e40 * ones(1,T); 
        return;
    end

    logdetR = 2*sum(log(diag(Lr)));
    invR = (Lr'\(Lr\eye(N)));

    h = zeros(T,N);
    h(1,:) = var(x);      

    for t = 2:T
        h(t,:) = omega' + alpha'.*(x(t-1,:).^2) + beta'.*h(t-1,:);
    end

    const = N*log(2*pi);

    LL = zeros(T,1);
    for t = 1:T
        ht = h(t,:)';
        if any(ht <= 0)
        LL = -1e40 * ones(1,T); 
            return;
        end

        Dt_inv = diag(1./sqrt(ht));      
        zt = Dt_inv * x(t,:)';           

        quad = zt' * invR * zt;
        logdetH = logdetR + sum(log(ht)); 

        LL(t) = -0.5*(const + logdetH + quad);
    end
    LL = LL(:);
end

function v = corr_vech(R)
    N = size(R,1);
    idx = find(tril(true(N),-1));
    rho = R(idx);
    rho = max(min(rho, 0.999), -0.999);
    v = atanh(rho);  
end


function R = vech_corr(v, N)
    idx = find(tril(true(N),-1));
    rho = tanh(v);
    R = eye(N);
    R(idx) = rho;
    R = R + R' - eye(N);
end

function [omega, alpha, beta, R] = unpack_ccc_params(theta, N)
    theta = theta(:);
    omega = theta(1:N);
    alpha = theta(N+1:2*N);
    beta  = theta(2*N+1:3*N);
    vCorr = theta(3*N+1:end);
    R = vech_corr(vCorr, N);

    R = (R + R')/2;
    R(1:N+1:end) = 1;
end

function LL = cccmvgarchLL_t(theta, x)

    [T, N] = size(x);

    [omega, alpha, beta, R, nu] = unpack_ccc_params_t(theta, N);

    if any(~isfinite([omega; alpha; beta; nu])) || nu <= 2
        LL = -1e40 * ones(1, T);
        return;
    end

    if any(omega <= 0) || any(alpha <= 0) || any(beta <= 0) || any(alpha + beta >= 1)
        LL = -1e40 * ones(1, T);
        return;
    end

    if any(abs(R(:)) > 1.000) || any(abs(diag(R) - 1) > 1e-6)
        LL = -1e40 * ones(1, T);
        return;
    end

    R = (R + R')/2;
    [Lr, p] = chol(R, 'lower');
    if p > 0
        LL = -1e40 * ones(1, T);
        return;
    end
    logdetR = 2 * sum(log(diag(Lr)));

    h = zeros(T, N);
    h(1,:) = var(x);  

    for t = 2:T
        h(t,:) = omega(:)' + alpha(:)'.*(x(t-1,:).^2) + beta(:)'.*h(t-1,:);
    end

    if any(~isfinite(h(:))) || any(h(:) <= 0)
        LL = -1e40 * ones(1, T);
        return;
    end

    c0 = gammaln((nu + N)/2) - gammaln(nu/2) - (N/2) * log(nu*pi);

    LL = zeros(1, T);

    for t = 1:T
        ht = h(t,:)';        
        xt = x(t,:)';               

        zt = xt ./ sqrt(ht);        

        y = Lr \ zt;             
        q = y' * y;                 

        logdetH = logdetR + sum(log(ht));

        LL(t) = c0 - 0.5*logdetH - ((nu + N)/2) * log(1 + q/nu);
    end
end


function [omega, alpha, beta, R, nu] = unpack_ccc_params_t(theta, N)
    theta = theta(:);

    omega = theta(1:N);
    alpha = theta(N+1:2*N);
    beta  = theta(2*N+1:3*N);

    vCorr = theta(3*N+1:end-1);
    R = vech_corr(vCorr, N);

    nu_raw = theta(end);
    nu = 2 + nu_raw; 

    R = (R + R')/2;
    R(1:N+1:end) = 1;
end



function W = checkWhiteACF(x, alpha, p)
% p = estimated parameters
N = length(x);
%alpha is deciding confidence interval
[r,lags] = autocorr(x);
Q = N*(N+2)*sum(r(2:end).^2 ./ (N-(1:20)'));
chiValue = chi2inv(1-alpha,20-p);
if Q < chiValue
    fprintf('The residuals are deemed white according to the Ljung-Box test for ACF (as %5.2f < %5.2f).\n',Q,chiValue);
else
    fprintf('The residuals are not deemed white according to the Ljung-Box test for ACF (as %5.2f > %5.2f).\n',Q,chiValue);
end
W = Q<chiValue;
end


function LL = dcc_garch_ll(theta, x)
    [T, N] = size(x);
    [omega, alpha, beta, a, b] = unpack_dcc_garch(theta, N);

    if any(~isfinite([omega; alpha; beta])) || any(omega<=0) || any(alpha<0) || any(beta<0) || any(alpha+beta>=1) || ~isfinite(a) || ~isfinite(b) || a<0 || b<0 || (a+b)>=1
        LL = -1e40*ones(1,T);
        return;
    end

    h = zeros(T,N);
    h(1,:) = var(x);

    for t=2:T
        h(t,:) = omega' + alpha'.*(x(t-1,:).^2) + beta'.*h(t-1,:);
    end

    eps = x ./ sqrt(h);

    Qbar = (eps'*eps)/T;  % unconditional covariance of standardized residuals
    Qbar = (Qbar + Qbar')/2;

    Qt = Qbar;
    LL = zeros(1,T);

    const = N*log(2*pi);

    for t=1:T
        if t>=2
            et1 = eps(t-1,:)';
            Qt  = (1-a-b)*Qbar + a*(et1*et1') + b*Qt;
            Qt  = (Qt + Qt')/2;
        end

        dq = sqrt(diag(Qt));
        Dq_inv = diag(1./dq);
        Rt = Dq_inv * Qt * Dq_inv;
        Rt = (Rt + Rt')/2;
        Rt(1:N+1:end) = 1;

        [L,p] = chol(Rt,'lower');
        if p>0
             LL(:) = -1e40;
        end

        logdetRt = 2*sum(log(diag(L)));

        et = eps(t,:)';
        y  = L \ et;
        quad = y' * y;

        LL(t) = -0.5*(const + sum(log(h(t,:)')) + logdetRt + quad);
    end
end


function [omega, alpha, beta, a, b] = unpack_dcc_garch(theta, N)
    w  = theta(1:N);
    alpha_tilde  = theta(N+1:2*N);
    beta_tilde  = theta(2*N+1:3*N);
    gamma_tilde = theta(3*N+1);
    delta_tilde = theta(3*N+2);

    omega = exp(w);

    ealp = exp(alpha_tilde); 
    ebet = exp(beta_tilde);
    denom = 1 + ealp + ebet;
    alpha = ealp ./ denom;
    beta  = ebet ./ denom;

    egam = exp(gamma_tilde); 
    edel = exp(delta_tilde);
    denD = 1 + egam + edel;
    a = egam / denD;
    b = edel / denD;
end
