%% Filtering in state space models
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

addpath('/Users/khw/Documents/År 5/Finansiell Statistik/Labbar/Lab4')

%% Affine short rate model
load CIRstruct2;


data = cir.data;  
N = cir.Nobs;
dt = cir.dt;

T = [1/4, 1/2, 1, 2, 3, 5]; % maturities
x = T; % time to maturity 

a = cir.a;
b = cir.b;
sigma = cir.s;

gamma  = sqrt(a^2 + 2*sigma^2);         
exp_gx = exp(gamma .* x);              

B = 2*(exp_gx - 1) ./ ((gamma + a)*(exp_gx - 1) + 2*gamma); 

A = ((2*gamma .* exp((a+gamma).*(x/2))) ./ ((gamma + a).*(exp_gx - 1) + 2*gamma)).^(2*a*b/sigma^2); 


C = (B ./ T)';     % linear measurement equation       
D = -(1./T .* log(A))';   


phi = 1 - a*dt;  % Euler discretization        
const = a*b*dt;          
Q = sigma^2 * b * dt; 


x_filt = zeros(N,1); % filtered r_k
P_filt = zeros(N,1); % filtered variance

R = 1e-4 * eye(6); % measurement noise covariance
x = b; % start at long-run mean
P = 0.1; % initial uncertainty

for k = 1:N
    x_pred = phi * x + const;     
    P_pred = phi * P * phi + Q;    
    
    yk     = yhat(k,:).';            
    y_pred = D + C * x_pred;         
    
    innov  = yk - y_pred;            
    S      = C * P_pred * C' + R;    
    K      = P_pred * C' / S;       
    
    x = x_pred + K * innov;          
    P = (1 - K * C) * P_pred;     
    
    x_filt(k) = x;
    P_filt(k) = P;
end


t_grid = (0:N-1)' * dt;

figure;
plot(t_grid, data, 'k-', 'LineWidth', 1.2); hold on;
plot(t_grid, x_filt, 'r-', 'LineWidth', 1.2);
xlabel('Time (years)');
ylabel('Short rate');
legend('True r_t','Kalman estimate r_t','Location','best');
title('Kalman filter estimate of CIR short rate');
grid on;

%% Extra Real Data

clear; 

load FredData;   

Y = FredData;    
T = [1/4, 1/2, 1, 2, 3, 5];
dt = 1/12;              % monthly data

[Tobs, m] = size(Y);

theta0 = log([0.1; 0.05; 0.05; 1e-4]);  % log [a, b, sigma, lambda]

negloglik_fun = @(theta) kalman_negloglik_CIR(theta, Y, T, dt);

[theta_hat, fval] = fminsearch(negloglik_fun, theta0);

a_hat = exp(theta_hat(1));
b_hat = exp(theta_hat(2));
sigma_hat = exp(theta_hat(3));
lambda_hat = exp(theta_hat(4));

fprintf('Estimated a = %.4f\n', a_hat);
fprintf('Estimated b = %.4f\n', b_hat);
fprintf('Estimated sigma = %.4f\n', sigma_hat);
fprintf('Estimated lambda = %.6f\n', lambda_hat);

[r_filt, ~] = kalman_filter_CIR(a_hat, b_hat, sigma_hat, lambda_hat, Y, T, dt);

time_grid = (0:Tobs-1)' * dt;
figure;
plot(time_grid, r_filt, 'b', 'LineWidth', 1.2);
xlabel('Time (years since Jan 2009)');
ylabel('Short rate');
title('Estimated short rate');
grid on;

%% Stochastic Volatility
clear;
load StochVol;

T = length(Z);

kappa0 = -2;
kappa1 = 0.7;
tau    = 1;

eta    = -1.2704;      
R      = (pi^2)/2;   

y = log(Z.^2) - eta;

x_filt = zeros(T,1);
P_filt = zeros(T,1);

x = kappa0 / (1 - kappa1);  % stationary mean of AR(1)
P = (tau^2)/(1-kappa1^2);   % stationary variance of x_t                  

Q = tau^2;                  

for t = 1:T
    x_pred = kappa0 + kappa1 * x;
    P_pred = kappa1^2 * P + Q;
    
    v = y(t) - x_pred;           
    S = P_pred + R;              
    K = P_pred / S;            
    
    x = x_pred + K * v;          
    P = (1 - K) * P_pred;      
    
    x_filt(t) = x;
    P_filt(t) = P;
end

sigma_hat = exp(0.5 * x_filt);


time = (1:T)';

figure;
plot(time, V, 'k-', 'LineWidth', 1.2); hold on;
plot(time, x_filt, 'r-', 'LineWidth', 1.2);
xlabel('t');
ylabel('\sigma_t');
legend('True log volatility','Kalman estimate','Location','best');
title('Kalman filter estimate of stochastic volatility');
grid on;

figure;
plot(time, sigma_hat, 'r-', 'LineWidth', 1.2);
xlabel('t'); ylabel('sigmahat_t');
title('Kalman-filtered volatility');
grid on;
%% The Particle Filter

N = 1000; 
n = length(y);
xhat = zeros(n,1);      
sigmahat = zeros(n,1);     

m0 = kappa0/(1 - kappa1);
v0 = tau^2/(1 - kappa1^2);
part = m0 + sqrt(v0) * randn(N,1);   
w = ones(N,1) / N;

obs_n = @(x,y) normpdf(y,x,S);
w = obs_n(part, y(1));
w = w / sum(w);                      
xhat(1) = sum(part .* w);          
sigmahat(1) = exp(0.5 * xhat(1));

ind  = randsample(N, N, true, w);
part = part(ind);

for t = 2:n
    part = kappa0 + kappa1 * part + tau * randn(N,1);
    w = obs_n(part, y(t));
    
    if sum(w) == 0
        w = ones(N,1) / N;
    else
        w = w / sum(w);
    end
    
    xhat(t) = sum(part .* w);
    sigmahat(t) = exp(0.5 * xhat(t));
    
    ind  = randsample(N, N, true, w);
    part = part(ind);
end

figure;
plot(sigmahat,'LineWidth',1.2)
title('Filtered Volatility (Particle Filter)');
xlabel('t'); ylabel('\sigma_t');
grid on;

%% The particle filter with t distribution 

nu = 4; % ska den estimeras? 

n = length(y);
xhat_t = zeros(n,1);
sigmahat_t = zeros(n,1);

m0 = kappa0 / (1 - kappa1);
v0 = tau^2 / (1 - kappa1^2);
part = m0 + sqrt(v0) * randn(N,1);  
w = ones(N,1) / N;

obs_t = @(x,y,nu) (tpdf(y ./ exp(0.5*x), nu) ./ exp(0.5*x));

w = obs_t(part, y(1), nu);
if sum(w) == 0
    w = ones(N,1) / N;
else
    w = w / sum(w);
end
xhat_t(1) = sum(part .* w);
sigmahat_t(1) = exp(0.5 * xhat_t(1));

ind = randsample(N, N, true, w);
part = part(ind);

for t = 2:n
    part = kappa0 + kappa1 * part + tau * randn(N,1);
    
    w = obs_t(part, y(t), nu);
    if sum(w) == 0
        w = ones(N,1) / N;
    else
        w = w / sum(w);
    end
    
    xhat_t(t) = sum(part .* w);
    sigmahat_t(t) = exp(0.5 * xhat_t(t));
    
    ind = randsample(N, N, true, w);
    part = part(ind);
end

figure;
plot(sigmahat_t,'LineWidth',1.2)
title('Filtered Volatility (Particle Filter)');
xlabel('t'); ylabel('\sigma_t');
grid on;
%% Example
% example of particle bootstrap filter
N = 1000; % number of particles
n = 60;   % number of observations
tau = zeros(1,n+1); % vector of filter means
w = zeros(N,1);    % vector of weights
p = @(x,y) normpdf(y,x,S); % observation density, for weights 
part = R*sqrt(1/(1 - A^2))*randn(N,1); % initialization
w = p(part,Y(1));   % weighting
tau(1) = sum(part.*w)/sum(w);  % estimation 
ind = randsample(N,N,true,w); % selection
part = part(ind);
for k = 1:n, % main loop
    part = A*part + R*randn(N,1); % mutation (move accord to AR(1) dyn
    w = p(part,Y(k + 1)); % weighting w. meas pdf
    tau(k + 1) = sum(part.*w)/sum(w); % estimation (calc cond exp)
    ind = randsample(N,N,true,w); % selection (resampling)
    part = part(ind);
end


%% Functions 

function nll = kalman_negloglik_CIR(theta, Y, T, dt)

    a     = exp(theta(1));
    b     = exp(theta(2));
    sigma = exp(theta(3));
    lam   = exp(theta(4));  

    [N, m] = size(Y);

    x = T;                            
    gamma  = sqrt(a^2 + 2*sigma^2);

    exp_gx = exp(gamma .* x);          

    B = 2*(exp_gx - 1) ./ ((gamma + a)*(exp_gx - 1) + 2*gamma );
    A = ((2*gamma .* exp((a+gamma).*(x/2))) ./ ((gamma + a).*(exp_gx - 1) + 2*gamma)) .^(2*a*b/sigma^2);

    C = (B ./ T)';                        
    D = -(1./T .* log(A))';                

    R = lam * eye(m);                      

    phi = 1 - a*dt;
    c_state = a*b*dt;
    Q = sigma^2 * b * dt;           

    x = b;                 
    P = 0.01;            

    loglik = 0;

    const = m * log(2*pi);

    for k = 1:N
        x_pred = phi * x + c_state;
        P_pred = phi * P * phi + Q;

        yk = Y(k,:).';              

        y_pred = D + C * x_pred;

        innov = yk - y_pred;        
        S = C * P_pred * C' + R;  

        [L,p] = chol(S,'lower');
        if p > 0
            nll = 1e10;
            return;
        end
        logdetS = 2*sum(log(diag(L))); % square of triangular 
        quadform = innov' * (S \ innov);

        loglik = loglik - 0.5 * (const + logdetS + quadform);

        K = P_pred * C' / S;              

        x = x_pred + K * innov;           
        P = (1 - K*C) * P_pred;         
    end

    nll = -loglik;   
end


function [x_filt, P_filt] = kalman_filter_CIR(a, b, sigma, lam, Y, T, dt)

    [N, m] = size(Y);

    x_filt = zeros(N,1);
    P_filt = zeros(N,1);

    x = T;
    gamma  = sqrt(a^2 + 2*sigma^2);
    exp_gx = exp(gamma .* x);

    B = 2*(exp_gx - 1) ./ ( (gamma + a)*(exp_gx - 1) + 2*gamma );
    A = ( (2*gamma .* exp((a+gamma).*(x/2))) ./ ((gamma + a).*(exp_gx - 1) + 2*gamma)) .^(2*a*b/sigma^2);

    C = (B ./ T)';                         
    D = -(1./T .* log(A))';                

    R = lam * eye(m);

    phi     = 1 - a*dt;
    c_state = a*b*dt;
    Q       = sigma^2 * b * dt;

    x = b;
    P = 0.01;

    for k = 1:N
        x_pred = phi * x + c_state;
        P_pred = phi * P * phi + Q;

        yk = Y(k,:).';
        y_pred = D + C * x_pred;

        innov = yk - y_pred;
        S = C * P_pred * C' + R;
        K = P_pred * C' / S;

        x = x_pred + K * innov;
        P = (1 - K*C) * P_pred;

        x_filt(k) = x;
        P_filt(k) = P;
    end
end

