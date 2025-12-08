%% Variance models
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

load garchdata.mat
%% GARCH-processes

omega = 0.1;
alpha = 0.25;
beta = 0.6;
mu = 8;

x = garchdata;

x0 = [mean(x); omega; alpha; beta];
%x0 = [5; 0.3; 0.3; 0.3];

[xout,logL,CovM]=MLmax(@garchLL,x0,x); 

SE = sqrt(diag(CovM));

CI_low  = xout - 1.96*SE;
CI_high = xout + 1.96*SE;


fprintf('Mu    = %8.4f | Estimated mu    = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', mu, xout(1), CI_low(1), CI_high(1));
fprintf('Omega = %8.4f | Estimated omega = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', omega, xout(2), CI_low(2), CI_high(2));
fprintf('Alpha = %8.4f | Estimated alpha = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', alpha, xout(3), CI_low(3), CI_high(3));
fprintf('Beta  = %8.4f | Estimated beta  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', beta, xout(4), CI_low(4), CI_high(4));

%% ARMA comparison 

epshat = x - mu;        
Y = epshat.^2; 
Y = Y(:);

arma = iddata(Y, [0*Y+1]);

SYS = armax(arma,[1 1 1 0]);

present(SYS);

a1 = -SYS.A(2);  
c1 =  SYS.C(2);    

a1garch = xout(3) + xout(4);
c1garch = -xout(4);

fprintf('AR(1):   ARMA = %8.4f   |   GARCH = %8.4f\n', a1, a1garch);
fprintf('MA(1):   ARMA = %8.4f   |   GARCH = %8.4f\n', c1, c1garch);

%% EGARCH-processes

load egarchdata.mat

ex = egarchdata;

omegae = 0.1;
alphae = 0.25;
betae = 0.6;
c = 0;
mu = 8;

ex0 = [mean(ex); omegae; alphae; betae];


[exout,elogL,eCovM]=MLmax(@EGARCHLL,ex0,ex);

SE = sqrt(diag(eCovM));

eCI_low  = exout - 1.96*SE;
eCI_high = exout + 1.96*SE;


fprintf('Mu    = %8.4f | Estimated mu    = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', mu, exout(1), eCI_low(1), eCI_high(1));
fprintf('Omega = %8.4f | Estimated omega = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', omegae, exout(2), eCI_low(2), eCI_high(2));
fprintf('Alpha = %8.4f | Estimated alpha = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', alphae, exout(3), eCI_low(3), eCI_high(3));
fprintf('Beta  = %8.4f | Estimated beta  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', betae, exout(4), eCI_low(4), eCI_high(4));


%[LL, LLS, ht] = egarch_likelihood(parameters, data, p, o, q, error_type, back_cast, T, estim_flag); % från toolbox ?

%% Multivariate GARCH models and market data
rho = 0.45;

P = [1 rho; rho 1];

T = 30000; 

tarch1 = [xout(2); xout(3); xout(4)]; 
tarch2 = [xout(2); xout(3); xout(4)]; 

param = [tarch1; tarch2; corr_vech(P)];


[data, ~, ~] = ccc_mvgarch_simulate(T, 2, param, 1, 0, 1);

theta = [tarch1; tarch2; rho];


[cccxout,ccclogL,cccCovM]=MLmax(@cccmvgarchLL, theta, data);

SE = sqrt(diag(cccCovM));

cccCI_low  = cccxout - 1.96*SE;
cccCI_high = cccxout + 1.96*SE;

disp(T);
fprintf('Omega = %8.4f | Estimated omega1 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', omega, cccxout(1), cccCI_low(1), cccCI_high(1));
fprintf('Alpha = %8.4f | Estimated alpha1 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', alpha, cccxout(2), cccCI_low(2), cccCI_high(2));
fprintf('Beta  = %8.4f | Estimated beta1  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', beta, cccxout(3), cccCI_low(3), cccCI_high(3));
fprintf('Omega = %8.4f | Estimated omega2 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', omega, cccxout(4), cccCI_low(4), cccCI_high(4));
fprintf('Alpha = %8.4f | Estimated alpha2 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', alpha, cccxout(5), cccCI_low(5), cccCI_high(5));
fprintf('Beta  = %8.4f | Estimated beta2  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', beta, cccxout(6), cccCI_low(6), cccCI_high(6));
fprintf('Rho  = %8.4f | Estimated rho  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', rho, cccxout(7), cccCI_low(7), cccCI_high(7));


%% Plotting SwedishStockData

load SwedishStockData.mat


dates_raw = X(:,1);       
stocks     = X(:,2:9);   
dates = datetime(num2str(dates_raw),'InputFormat','yyyyMMdd');

figure; hold on;
for i = 1:8
    plot(dates, stocks(:,i), 'LineWidth', 1.2)
end
hold off;

legend(names, 'Location', 'best', 'Interpreter', 'none')
xlabel('Date')
ylabel('Price / Return')
title('Swedish Stock Series')
grid on

nordea = X(:,8);
investor = X(:,5);
figure; hold on;
    plot(nordea, 'LineWidth', 1.2)
    plot(investor, 'LineWidth', 1.2)
hold off;

figure;
subplot(2,1,1);
normplot(nordea);
subplot(2,1,2);
normplot(investor);

%% Fitting the ccc
% Investor and Nordea 

nordea = nordea- mean(nordea);
investor = investor - mean(investor);
theta = [tarch1; tarch2; rho];

% 
% [nxout,a,~]=MLmax(@garchLL,[1 0.25 0.25 0.25],nordea); 
% [ixout,~,~]=MLmax(@garchLL,[1 0.25 0.25 0.25],investor); 
% theta = [nxout(2:4); ixout(2:4); rho];

stockdata = [nordea investor];

[stocksxout,stockslogL,stocksCovM]=MLmax(@cccmvgarchLL, theta, stockdata);

SE = sqrt(diag(stocksCovM));

stocksCI_low  = stocksxout - 1.96*SE;
stocksCI_high = stocksxout + 1.96*SE;


fprintf(' Estimated omega1 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksxout(1), stocksCI_low(1), stocksCI_high(1));
fprintf('Estimated alpha1 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksCI_low(2), stocksCI_low(2), stocksCI_high(2));
fprintf('Estimated beta1  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksCI_low(3), stocksCI_low(3), stocksCI_high(3));
fprintf('Estimated omega2 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksxout(4), stocksCI_low(4), stocksCI_high(4));
fprintf('Estimated alpha2 = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksCI_low(5), stocksCI_low(5), stocksCI_high(5));
fprintf('Estimated beta2  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksCI_low(6), stocksCI_low(6), stocksCI_high(6));
fprintf('Estimated rho  = %8.4f | 95%% CI: [%8.4f , %8.4f]\n', stocksCI_low(7), stocksCI_low(7), stocksCI_high(7));



%% Functions

function LL = garchLL(theta, x)
    x= x(:);
    mu = theta(1);
    omega = theta(2);
    alpha = theta(3);
    beta = theta(4);

    T = length(x);
    eps = x - mu;

    if omega <= 0 || alpha <= 0 || beta <= 0 || (alpha + beta >= 1)
        LL = -1e20 * ones(1,T);   
        return;
    end

    sigma2 = zeros(T,1);
    sigma2(1) = var(eps);

    for t = 2:T
        sigma2(t) = omega + alpha*eps(t-1)^2 + beta*sigma2(t-1);
        if sigma2(t) <= 0
            LL = -1e20 * ones(1,T);
            return;
        end
    end

    LL = -0.5 * (log(2*pi) + log(sigma2) + eps.^2 ./ sigma2 ); % assuming normal distribution 
end

function LL = EGARCHLL(theta, x)
    x = x(:);
    mu = theta(1);
    omega = theta(2);
    alpha = theta(3);
    beta = theta(4);

    T = length(x);
    eps = x - mu;

    logsigma2 = zeros(T,1);
    logsigma2(1) = log(var(eps));

    for t = 2:T
        logsigma2(t) = omega + alpha*abs(eps(t-1)) + beta*logsigma2(t-1); % c=0
    end

    LL = -0.5 * (log(2*pi) + logsigma2 + eps.^2 ./ exp(logsigma2) ); % also assuming normal distribution
end




function LL = cccmvgarchLL(theta, x)
    [T, ~] = size(x);

    omega1 = theta(1);
    alpha1 = theta(2);
    beta1 = theta(3);
    omega2 = theta(4);
    alpha2 = theta(5);
    beta2 = theta(6);

    rho = theta(7);
    R = [1   rho;
         rho 1  ];


    invR = inv(R);

    r1 = x(:,1);
    r2 = x(:,2);

    h = zeros(T,2);
    h(1,1) = var(r1);   
    h(1,2) = var(r2);  

    if omega1 <= 0 || alpha1 <= 0 || beta1 <= 0 || (alpha1 + beta1 >= 1)
        LL = -1e40 * ones(1,T);   
        return;
    end
    if omega2 <= 0 || alpha2 <= 0 || beta2 <= 0 || (alpha2 + beta2 >= 1)
        LL = -1e40 * ones(1,T);   
        return;
    end

    if -1 >rho || rho > 1
         LL = -1e40 * ones(1,T);   
        return;
    end 

    for t = 2:T
        h(t,1) = omega1 + alpha1 * r1(t-1)^2 + beta1 * h(t-1,1);
        h(t,2) = omega2 + alpha2 * r2(t-1)^2 + beta2 * h(t-1,2);
    end


    LL = zeros(T,1);

   logdetR = log(det(R));  
   const = log(2*pi);

    for t = 1:T
        h1t2 = h(t,1);
        h2t2 = h(t,2);
        
        D = diag([sqrt(h1t2), sqrt(h2t2)]);
        rt = [r1(t); r2(t)];
        
        z = inv(D)*rt;

        quadform = z' * invR * z;   
        logHt = logdetR + log(h1t2) + log(h2t2);
        LL(t) = -0.5 * ( const + logHt + quadform ); 
    end
    LL = LL(:);
end
