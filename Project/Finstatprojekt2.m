%% Financial Statistics Project Part 2
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

%% 2: Calibration of stock options, track A
clear;
load('OMXS30optA.mat');
N = 49;
N_training = 49;
x0 = [0.01 0.5 0.5 0.01 -0.99]; 
optData = opt(4);
prices4 = opt_price('Heston',x0,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
optData = opt(3);
prices3 = opt_price('Heston',x0,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
figure;
hold on; grid on;
plot(prices4, 'LineWidth', 1.2);
plot(prices3, 'LineWidth', 1.2);
xlabel('Option index');
ylabel('Heston model option price');
title('Heston model option prices for two option maturities');
legend({'Option set 4', 'Option set 3'}, ...
       'Location', 'best');
hold off;

%% Parameter estimations
% OLS BS
% PLS BS
% OLS Heston
% PLS Heston
% OLS Bates
% PLS Bates
% OLS CGMY
% PLS CGMY

paramNamesBS = {'\sigma'};
paramNamesH = {'\kappa','\theta','\sigma','v_0','\rho'};
paramNamesB = {'V_0', '\kappa', '\theta', '\sigma', '\rho', '\lambda', '\mu_J', '\sigma_J'};
paramNamesCGMY = {'C', 'G', 'M', 'Y'};
lambda = 5;

%% OLS BS 
firstFunc = @(par) ordinaryLeastSquaresBS(par,opt(1));
x0 = [0.5];
lb = [0];
ub = [inf];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
x_resultBSOLS = zeros(N_training,1);
optTest = zeros(N_training,1);
opt_prices_BSOLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    func = @(par) ordinaryLeastSquaresBS(par,optData);
        if n == 1
            start = initial_x;
        else
           start = x_resultBSOLS(n-1);
            if ~isfinite(start) || start <= 0
                start = initial_x;
            end
        end
     x_resultBSOLS(n) = lsqnonlin(func, start, lb, ub);
     optTest(n) = mean(optData.S);
     opt_prices_BSOLS{n} = opt_price('BS',start,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
end
figure;
plot(1:N_training, x_resultBSOLS, 'LineWidth', 1.2);
grid on;
xlabel('Time index');
ylabel('\sigma_{BS}');
title('Estimated Black-Scholes implied volatility over time (OLS)');
%% PLS BS 
firstFunc = @(par) ordinaryLeastSquaresBS(par,opt(1)); % using OLS for first
x0 = [0.5];
lb = [0];
ub = [inf];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
parPrev = initial_x;
x_resultBSPLS = zeros(N_training,1);
optTest = zeros(N_training,1);
opt_prices_BSPLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    funcPLS = @(par) penalizedLeastSquaresBS(par, optData, parPrev, lambda); 
        if n == 1
            x_resultBSPLS(n) = lsqnonlin(funcPLS, initial_x, lb, ub);
            opt_prices_BSPLS{n} = opt_price('BS',x_resultBSPLS(n),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          parPrev = x_resultBSPLS(n-1);
          if (~isfinite(parPrev) || parPrev <= 0)
              parPrev = initial_x;
          end
          funcPLS = @(par) penalizedLeastSquaresBS(par, optData, parPrev, lambda);
          x_resultBSPLS(n) = lsqnonlin(funcPLS, parPrev,lb, ub);
          opt_prices_BSPLS{n} = opt_price('BS',x_resultBSPLS(n-1),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);
end
figure;
plot(1:N_training, x_resultBSPLS, 'LineWidth', 1.2);
grid on;
xlabel('Time index');
ylabel('\sigma_{BS}');
title('Estimated Black-Scholes implied volatility over time (PLS)');
%% OLS Heston
firstFunc = @(par) ordinaryLeastSquares(par,opt(1));
x0 = [0.01 0.5 0.5 0.01 -0.99];
lb = [0 0 0 0 -1];
ub = [inf inf inf inf 1];
intitial_x = lsqnonlin(firstFunc,x0, lb, ub);
x_result = zeros(N_training,5);
optTest = zeros(N_training,1);
opt_prices_HOLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    func = @(par) ordinaryLeastSquares(par,optData);
        if n == 1
            x_result(n,:) = lsqnonlin(func, intitial_x, lb, ub);
            opt_prices_HOLS{n} = opt_price('Heston',x_result(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          start = x_result(n-1,:);
          if any(~isfinite(start)) || any(start(1:4) <= 0) || abs(start(5)) >= 0.99
              x_result(n,:) = lsqnonlin(func, intitial_x, lb, ub);
          else 
              x_result(n,:) = lsqnonlin(func, x_result(n-1,:),lb, ub );
          end
          opt_prices_HOLS{n} = opt_price('Heston',x_result(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);
end

figure;
for i = 1:5
    subplot(3,2,i);
    plot(x_result(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesH{i});
    xlabel('Time index');
end
sgtitle('Estimated Heston parameters over time with OLS');
%% PLS Heston
firstFunc = @(par) ordinaryLeastSquares(par,opt(1)); % using OLS for first
x0 = [0.01 0.5 0.5 0.01 -0.99];
lb = [0 0 0 0 -1];
ub = [inf inf inf inf 1];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
parPrev = initial_x;
x_resultPLS = zeros(N_training,5);
optTest = zeros(N_training,1);
opt_prices_HPLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    funcPLS = @(par) penalizedLeastSquares(par, optData, parPrev, lambda); 
        if n == 1
            x_resultPLS(n,:) = lsqnonlin(funcPLS, initial_x, lb, ub);
            opt_prices_HPLS{n} = opt_price('Heston',x_resultPLS(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          parPrev = x_resultPLS(n-1,:);
          if any(~isfinite(parPrev)) || any(parPrev(1:4) <= 0) || abs(parPrev(5)) >= 0.99
              parPrev = initial_x;
          end
          funcPLS = @(par) penalizedLeastSquares(par, optData, parPrev, lambda);
          x_resultPLS(n,:) = lsqnonlin(funcPLS, parPrev,lb, ub);
          opt_prices_HPLS{n} = opt_price('Heston',x_resultPLS(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);
end

figure;
for i = 1:5
    subplot(3,2,i);
    plot(x_resultPLS(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesH{i});
    xlabel('Time index');
end
sgtitle('Estimated Heston parameters over time with PLS');
%% OLS Bates
firstFunc = @(par) ordinaryLeastSquaresB(par,opt(1));
x0 = [0.01 0.5 0.5 0.01 -0.9 0.3 0.3 0.3];
lb = [0 0 0 0 -1 0 -inf 0];
ub = [inf inf inf inf 1 inf inf inf];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
x_resultBOLS = zeros(N_training,8);
optTest = zeros(N_training,1);
opt_prices_BOLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    func = @(par) ordinaryLeastSquaresB(par,optData);
        if n == 1
            x_resultBOLS(n,:) = lsqnonlin(func, initial_x, lb, ub);
            opt_prices_BOLS{n} = opt_price('Bates',x_resultBOLS(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          start = x_resultBOLS(n-1,:);
          if any(~isfinite(start)) || any(start(1:4) <= 0) || abs(start(5)) >= 0.99 || start(6) <= 0 || start(8) <= 0
              x_resultBOLS(n,:) = lsqnonlin(func, initial_x, lb, ub);
          else 
              x_resultBOLS(n,:) = lsqnonlin(func, x_resultBOLS(n-1,:),lb, ub );
          end
          opt_prices_BOLS{n} = opt_price('Bates',x_resultBOLS(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);
end
figure;
for i = 1:8
    subplot(4,2,i); 
    plot(x_resultBOLS(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesB{i});
    xlabel('Time index');
end
sgtitle('Estimated Bates parameters over time with OLS');
%% PLS Bates
firstFunc = @(par) ordinaryLeastSquaresB(par,opt(1)); % using OLS for first
x0 = [0.01 0.5 0.5 0.01 -0.9 0.3 0.3 0.3];
lb = [0 0 0 0 -1 0 -inf 0];
ub = [inf inf inf inf 1 inf inf inf];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
parPrev = initial_x;
x_resultBPLS = zeros(N_training,8);
optTest = zeros(N_training,1);
opt_prices_BPLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    funcPLS = @(par) penalizedLeastSquaresB(par, optData, parPrev, lambda); 
        if n == 1
            x_resultBPLS(n,:) = lsqnonlin(funcPLS, initial_x, lb, ub);
            opt_prices_BPLS{n} = opt_price('Bates',x_resultBPLS(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          parPrev = x_resultBPLS(n-1,:);
          if any(~isfinite(parPrev)) || any(parPrev(1:4) <= 0) || abs(parPrev(5)) >= 0.99 || parPrev(6) <= 0 || parPrev(8) <= 0
              parPrev = initial_x;
          end
          funcPLS = @(par) penalizedLeastSquaresB(par, optData, parPrev, lambda);
          x_resultBPLS(n,:) = lsqnonlin(funcPLS, parPrev,lb, ub);
          opt_prices_BPLS{n} = opt_price('Bates',x_resultBPLS(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);

end
figure;
for i = 1:8
    subplot(4,2,i); 
    plot(x_resultBPLS(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesB{i});
    xlabel('Time index');
end
sgtitle('Estimated Bates parameters over time with PLS');
%% OLS CGMY
firstFunc = @(par) ordinaryLeastSquaresC(par,opt(1));
x0 = [0.1 0.5 2 1];
lb = [0 0 1 0];
ub = [inf inf inf 2];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
x_resultCOLS = zeros(N_training,4);
optTest = zeros(N_training,1);
opt_prices_COLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    func = @(par) ordinaryLeastSquaresC(par,optData);
        if n == 1
            x_resultCOLS(n,:) = lsqnonlin(func, initial_x, lb, ub);
            opt_prices_COLS{n} = opt_price('CGMY',x_resultCOLS(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        else
          start = x_resultCOLS(n-1,:);
          if any(~isfinite(start)) || start(1) <= 0 || start(2) < 0  || start(3) < 1 ||  abs(start(4)-1) > 1 
              x_resultCOLS(n,:) = lsqnonlin(func, initial_x, lb, ub);
          else 
              x_resultCOLS(n,:) = lsqnonlin(func, x_resultCOLS(n-1,:),lb, ub );
          end
          opt_prices_COLS{n} = opt_price('CGMY',x_resultCOLS(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);
end
figure;
for i = 1:4
    subplot(2,2,i); 
    plot(x_resultCOLS(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesCGMY{i});
    xlabel('Time index');
end
sgtitle('Estimated CGMY parameters over time with OLS');
%% PLS CGMY
firstFunc = @(par) ordinaryLeastSquaresC(par,opt(1)); % using OLS for first
x0 = [0.1 0.5 2 1];
lb = [0 0 1 0];
ub = [inf inf inf 2];
initial_x = lsqnonlin(firstFunc,x0, lb, ub);
parPrev = initial_x;
x_resultCPLS = zeros(N_training,4);
optTest = zeros(N_training,1);
opt_prices_CPLS = cell(1,N_training);
for n = 1:N_training
    optData = opt(n);
    funcPLS = @(par) penalizedLeastSquaresC(par, optData, parPrev, lambda); 
        if n == 1
            x_resultCPLS(n,:) = lsqnonlin(funcPLS, initial_x, lb, ub);
            opt_prices_CPLS{n} = opt_price('CGMY',x_resultCPLS(n,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');

        else
          parPrev = x_resultCPLS(n-1,:);
          if any(~isfinite(parPrev)) || parPrev(1) <= 0 || parPrev(2) < 0  || parPrev(3) < 1 ||  abs(parPrev(4)-1) > 1 
              parPrev = initial_x;
          end
          funcPLS = @(par) penalizedLeastSquaresC(par, optData, parPrev, lambda);
          x_resultCPLS(n,:) = lsqnonlin(funcPLS, parPrev,lb, ub);
          opt_prices_CPLS{n} = opt_price('CGMY',x_resultCPLS(n-1,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
        end
    optTest(n) = mean(optData.S);

end
figure;
for i = 1:4
    subplot(2,2,i); 
    plot(x_resultCPLS(:,i),'LineWidth',1.2);
    grid on;
    title(paramNamesCGMY{i});
    xlabel('Time index');
end
sgtitle('Estimated CGMY parameters over time with PLS');
%% Comparison Plots
% BS
% figure;
% hold on; grid on;
% plot(1:N_training,x_resultBSPLS, 'r-', 'LineWidth', 0.7);
% plot(1:N_training,x_resultBSOLS,    'b--', 'LineWidth', 0.7);
% title(paramNamesBS);
% xlabel('Time index');
% 
% 
% legend({'PLS','OLS'});
% hold off;
% sgtitle('Estimated Black-Scholes parameters over time: PLS vs OLS');
% Heston
% figure;
% for i = 1:5
%     subplot(3,2,i); hold on; grid on;
% 
%     plot(x_resultPLS(:,i), 'r-', 'LineWidth', 0.7);
%     plot(x_result(:,i),    'b-', 'LineWidth', 0.7);
% 
%     title(paramNamesH{i});
%     xlabel('Time index');
% 
%     if i == 1
%         legend({'PLS','OLS'}, 'Location', 'best');
%     end
% 
%     hold off;
% end
% sgtitle('Estimated Heston parameters over time: PLS vs OLS');
% Bates
% figure;
% for i = 1:8
%     subplot(4,2,i); hold on; grid on;
% 
%     plot(x_resultBPLS(:,i), 'r-', 'LineWidth', 0.7);
%     plot(x_resultBOLS(:,i),    'b-', 'LineWidth', 0.7);
% 
%     title(paramNamesB{i});
%     xlabel('Time index');
% 
%     if i == 1
%         legend({'PLS','OLS'});
%     end
% 
%     hold off;
% end
% sgtitle('Estimated Bates parameters over time: PLS vs OLS');
% CGMY
% figure;
% for i = 1:4
%     subplot(2,2,i); hold on; grid on;
% 
%     plot(x_resultCPLS(:,i), 'r-', 'LineWidth', 0.7);
%     plot(x_resultCOLS(:,i),    'b-', 'LineWidth', 0.7);
% 
%     title(paramNamesCGMY{i});
%     xlabel('Time index');
% 
%     if i == 1
%         legend({'PLS','OLS'});
%     end
% 
%     hold off;
% end
% sgtitle('Estimated CGMY parameters over time: PLS vs OLS');

%% Predicting

%Choosing n=10 to be the choosen parameters since we choose 1-10 to be
%training data.

opt_prices_BSOLS2 = cell(1,N_training);
opt_prices_BSPLS2 = cell(1,N_training);
opt_prices_HOLS2 = cell(1,N_training);
opt_prices_HPLS2 = cell(1,N_training);
opt_prices_BOLS2 = cell(1,N_training);
opt_prices_BPLS2 = cell(1,N_training);
opt_prices_COLS2 = cell(1,N_training);
opt_prices_CPLS2 = cell(1,N_training);


for t = 1:N_training
    optData = opt(t);
    opt_prices_BSOLS2{t} = opt_price('BS',x_resultBSOLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_BSPLS2{t} = opt_price('BS',x_resultBSPLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_HOLS2{t} = opt_price('Heston',x_result(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_HPLS2{t} = opt_price('Heston',x_resultPLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_BOLS2{t} = opt_price('Bates',x_resultBOLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_BPLS2{t} = opt_price('Bates',x_resultBPLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_COLS2{t} = opt_price('CGMY',x_resultCOLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    opt_prices_CPLS2{t} = opt_price('CGMY',x_resultCPLS(10,:),optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
end


%% Tests errors in time dimension

[MAErrorBSOLS, MSErrorBSOLS, RMSErrorBSOLS] = computeErrors(opt,opt_prices_BSOLS2, 'BS with OLS', N_training);
[MAErrorBSPLS, MSErrorBSPLS, RMSErrorBSPLS] = computeErrors(opt,opt_prices_BSPLS2, 'BS with PLS', N_training);
[MAErrorHOLS, MSErrorHOLS, RMSErrorHOLS] = computeErrors(opt,opt_prices_HOLS2, 'Heston with OLS', N_training);
[MAErrorHPLS, MSErrorHPLS, RMSErrorHPLS] = computeErrors(opt,opt_prices_HPLS2, 'Heston with PLS', N_training);
[MAErrorBOLS, MSErrorBOLS, RMSErrorBOLS] = computeErrors(opt,opt_prices_BOLS2, 'Bates with OLS', N_training);
[MAErrorBPLS, MSErrorBPLS, RMSErrorBPLS] = computeErrors(opt,opt_prices_BPLS2, 'Bates with PLS', N_training);
[MAErrorCOLS, MSErrorCOLS, RMSErrorCOLS] = computeErrors(opt,opt_prices_COLS2, 'CGMY with OLS', N_training);
[MAErrorCPLS, MSErrorCPLS, RMSErrorCPLS] = computeErrors(opt,opt_prices_CPLS2, 'CGMY with PLS', N_training);


figure
subplot(2,2,1);
hold on
plot(MAErrorBSOLS)
plot(MAErrorBSPLS)
plot(MAErrorHOLS)
plot(MAErrorHPLS)
plot(MAErrorBOLS)
plot(MAErrorBPLS)
plot(MAErrorCOLS)
plot(MAErrorCPLS)
title('Mean absolute error')
xlabel('Time index')
ylabel('Mean absolute error')
legend('BS OLS','BS PLS','Heston OLS','Heston PLS','Bates OLS','Bates PLS','CGMY OLS','CGMY PLS')
hold off

subplot(2,2,2);
hold on
plot(MSErrorBSOLS)
plot(MSErrorBSPLS)
plot(MSErrorHOLS)
plot(MSErrorHPLS)
plot(MSErrorBOLS)
plot(MSErrorBPLS)
plot(MSErrorCOLS)
plot(MSErrorCPLS)
title('Mean squared error')
xlabel('Time index')
ylabel('Mean squared error')
hold off

subplot(2,2,3);
hold on
plot(RMSErrorBSOLS)
plot(RMSErrorBSPLS)
plot(RMSErrorHOLS)
plot(RMSErrorHPLS)
plot(RMSErrorBOLS)
plot(RMSErrorBPLS)
plot(RMSErrorCOLS)
plot(RMSErrorCPLS)
title('Root mean squared error')
xlabel('Time index')
ylabel('Root mean squared error')
hold off




%% Tests errors in maturity dimension

[T1,x1] = maturityDimension(opt, opt_prices_BSOLS2, 'BS with OLS', N_training);
[T2,x2] = maturityDimension(opt, opt_prices_BSPLS2, 'BS with PLS', N_training);
[T3,x3] = maturityDimension(opt, opt_prices_HOLS2, 'Heston with OLS', N_training);
[T4,x4] = maturityDimension(opt, opt_prices_HPLS2, 'Heston with PLS', N_training);
[T5,x5] = maturityDimension(opt, opt_prices_BOLS2, 'Bates with OLS', N_training);
[T6,x6] = maturityDimension(opt, opt_prices_BPLS2, 'Bates with PLS', N_training);
[T7,x7] = maturityDimension(opt, opt_prices_COLS2, 'CGMY with OLS', N_training);
[T8,x8] = maturityDimension(opt, opt_prices_CPLS2, 'CGMY with PLS', N_training);


figure;
hold on
grid on;
scatter(x1,T1,'filled');
scatter(x2,T2,'filled');
scatter(x3,T3,'filled');
scatter(x4,T4,'filled');
scatter(x5,T5,'filled');
scatter(x6,T6,'filled');
scatter(x7,T7,'filled');
scatter(x8,T8,'filled');
xlabel('Time to Maturity');
ylabel('Mean Absolute Error');
title('Mean absolute error per Time to Maturity');
legend('BS OLS','BS PLS','Heston OLS','Heston PLS','Bates OLS','Bates PLS','CGMY OLS','CGMY PLS')
hold off


%% Tests errors in strike dimension


[T1,x1] = strikeDimension(opt, opt_prices_BSOLS2, 'BS with OLS', N_training);
[T2,x2] = strikeDimension(opt, opt_prices_BSPLS2, 'BS with PLS', N_training);
[T3,x3] = strikeDimension(opt, opt_prices_HOLS2, 'Heston with OLS', N_training);
[T4,x4] = strikeDimension(opt, opt_prices_HPLS2, 'Heston with PLS', N_training);
[T5,x5] = strikeDimension(opt, opt_prices_BOLS2, 'Bates with OLS', N_training);
[T6,x6] = strikeDimension(opt, opt_prices_BPLS2, 'Bates with PLS', N_training);
[T7,x7] = strikeDimension(opt, opt_prices_COLS2, 'CGMY with OLS', N_training);
[T8,x8] = strikeDimension(opt, opt_prices_CPLS2, 'CGMY with PLS', N_training);


figure;
hold on
grid on;
scatter(x1,T1,'filled');
scatter(x2,T2,'filled');
scatter(x3,T3,'filled');
scatter(x4,T4,'filled');
scatter(x5,T5,'filled');
scatter(x6,T6,'filled');
scatter(x7,T7,'filled');
scatter(x8,T8,'filled');
xlabel('Strike');
ylabel('Mean Absolute Error');
title('Mean absolute error per Strike');
legend('BS OLS','BS PLS','Heston OLS','Heston PLS','Bates OLS','Bates PLS','CGMY OLS','CGMY PLS')
hold off


%% Functions
    
function ols = ordinaryLeastSquaresBS(par,optData)
    if par(1) <= 0 
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    prices=opt_price('BS',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    prices = prices(:);
    mid = optData.MID(:);
    if any(~isfinite(prices(:)))
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    ols = prices-mid;
end

function pls = penalizedLeastSquaresBS(par,optData, parPrev, lambda)
    if par(1) <= 0 
        pls = 1e50* ones(numel(optData.MID)+length(par),1);
        return;
    end
    prices=opt_price('BS',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    if any(~isfinite(prices(:)))
        pls = 1e50* ones(numel(optData.MID)+length(par),1);
        return;
    end
    ols = ordinaryLeastSquaresBS(par,optData);
    resPenalty = sqrt(lambda) * (par(:) - parPrev(:)); 
    pls = [ols; resPenalty];
end

function ols = ordinaryLeastSquares(par,optData)
    if par(1) <= 0 || par(2) <= 0 || par(3) <= 0 || par(4) <= 0 || par(5) <= -1 || par(5) >= 1
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    prices=opt_price('Heston',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    prices = prices(:);
    mid = optData.MID(:);
   if any(~isfinite(prices(:)))
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    ols = prices-mid;
end

function pls = penalizedLeastSquares(par,optData, parPrev, lambda)
    if par(1) <= 0 || par(2) <= 0 || par(3) <= 0 || par(4) <= 0 || par(5) <= -1 || par(5) >= 1
        pls = 1e50* ones(numel(optData.MID)+length(par),1);
        return;
    end
    prices=opt_price('Heston',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    if any(~isfinite(prices(:)))
        pls = 1e50* ones(numel(optData.MID)+length(par),1);
        return;
    end
    ols = ordinaryLeastSquares(par,optData);
    resPenalty = sqrt(lambda) * (par(:) - parPrev(:)); 
    pls = [ols; resPenalty];
end

function ols = ordinaryLeastSquaresB(par,optData)
    if par(1) <= 0 || par(2) <= 0 || par(3) <= 0 || par(4) <= 0 || par(5) <= -1 || par(5) >= 1 || par(6) <= 0 || par(8)<=0
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    prices=opt_price('Bates',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
    prices = prices(:);
    mid = optData.MID(:);
    if any(~isfinite(prices(:)))
        ols = 1e50* ones(numel(optData.MID),1);
        return;
    end
    ols = prices-mid;
end

function pls = penalizedLeastSquaresB(par,optData, parPrev, lambda)
 if par(1) <= 0 || par(2) <= 0 || par(3) <= 0 || par(4) <= 0 || par(5) <= -1 || par(5) >= 1 || par(6) <= 0 || par(8)<=0
 pls = 1e50* ones(numel(optData.MID)+length(par),1);
 return;
 end
 prices=opt_price('Bates',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
 if any(~isfinite(prices(:)))
 pls = 1e50* ones(numel(optData.MID)+length(par),1);
 return;
 end
 ols = ordinaryLeastSquaresB(par,optData);
 resPenalty = sqrt(lambda) * (par(:) - parPrev(:)); 
 pls = [ols; resPenalty];
end

function ols = ordinaryLeastSquaresC(par,optData)
 if par(1) <= 0 || par(2) < 0 || par(3) < 1 || abs(par(4)-1) > 1 
 ols = 1e50* ones(numel(optData.MID),1);
 return;
 end
 prices=opt_price('CGMY',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
 prices = prices(:);
 mid = optData.MID(:);
 if any(~isfinite(prices(:)))
 ols = 1e50* ones(numel(optData.MID),1);
 return;
 end
 ols = prices-mid;
end

function pls = penalizedLeastSquaresC(par,optData, parPrev, lambda)
 if par(1) <= 0 || par(2) < 0 || par(3) < 1 || abs(par(4)-1) > 1 
 pls = 1e50* ones(numel(optData.MID)+length(par),1);
 return;
 end
 prices=opt_price('CGMY',par,optData.C',ones(size(optData.C))',optData.K',optData.S',optData.r',optData.T');
 if any(~isfinite(prices(:)))
 pls = 1e50* ones(numel(optData.MID)+length(par),1);
 return;
 end
 ols = ordinaryLeastSquaresC(par,optData);
 resPenalty = sqrt(lambda) * (par(:) - parPrev(:)); 
 pls = [ols; resPenalty];
end

% Assessment methods 
function e = MAE(y,y_pred)
    e = sum(abs(y-y_pred));
    e = e/length(y);
end

function e = MSE(y,y_pred)

e = sum((y-y_pred).^2);
e = e/length(y);

end

function e = RMSE(y,y_pred)

e = sum((y-y_pred).^2);
e = sqrt(e/length(y));

end

function [MAError, MSError, RMSError] = computeErrors(opt, opt_prices, modelName, N)
modelName = string(modelName);
cellError = cell(1,N);

for k = 1:N
    MAError(k) = MAE(opt(k).MID',opt_prices{k});
    MSError(k) = MSE(opt(k).MID',opt_prices{k});
    RMSError(k) = RMSE(opt(k).MID',opt_prices{k});
    cellError{k} = abs(opt(k).MID' - opt_prices{k});
end


end

function [T,x] = maturityDimension(opt, opt_prices, modelName,N)
values = cell(1,13);
pred_values = cell(1,13);
modelName = string(modelName);

for k = 1:13
    for t = 10:N
        indices = find(max(-0.05 + 0.10*(k-1),0) < opt(t).T & opt(t).T < -0.05 + 0.10*(k));
        if ~isempty(indices)
            x(k) = opt(t).T(indices(1));
        else
            x(k) = (max(-0.05 + 0.10*(k-1),0) + -0.05 + 0.10*(k))/2;
        end
        values{k} = [values{k}; opt(t).MID(indices)];
        pred_values{k} = [pred_values{k}, opt_prices{t}(indices)];
    end
    squareError(k) = MAE(values{k}',pred_values{k});

end

T = squareError;

figure;
scatter(x,squareError,'filled');
xlabel('Time to Maturity');
ylabel('Mean Absolute Error');
title('Mean absolute error per Time to Maturity for ' + modelName);

end

function [T,xK] = strikeDimension(opt, opt_prices, modelName, N)
valuesK = cell(1,45);
pred_valuesK = cell(1,45);
modelName = string(modelName);

for k = 1:45
    indicesALL = [];
    for t = 10:N

        indices = find(445 + 10*(k-1) < opt(t).K & opt(t).K < 445 + 10*(k));
        indicesALL = [indicesALL; indices];
        if ~isempty(indicesALL)
            xK(k) = opt(t).K(indicesALL(1));
        else
            xK(k) = (445 + 10*(k-1) + 445 + 10*(k))/2;
        end
        valuesK{k} = [valuesK{k}; opt(t).MID(indices)];

        pred_valuesK{k} = [pred_valuesK{k}, opt_prices{t}(indices)];
    end
    squareErrorK(k) = MAE(valuesK{k}',pred_valuesK{k});

end

T = squareErrorK;

figure;
scatter(xK,squareErrorK,'filled');
xlabel('Strike');
ylabel('Mean Absolute Error');
title('Mean absolute error per strike for' + modelName);


end
