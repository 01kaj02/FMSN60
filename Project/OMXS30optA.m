%% Financial Statistics Project
% Kajsa Hansson Willis & Victoria Lagerstedt
% FMSN60: Financial Statistics 2026
clear; clc;

addpath('/Users/khw/Documents/År 5/Finansiell Statistik/Projekt/Kod+data/')

 %% 1: Intra-day and daily data using standard GARCH-family models and GARCH-x models for volatility, track A
clear;
load('VOLV0A.mat') % 2003-01-02-2003-12-12 



%% 2: Calibration of stock options, track A
clear;
load('OMXS30optA.mat');

%opt_price('Bates',par,opt(1).C,ones(size(opt(1),C)),opt(1).K,opt(1).S,opt(1).r,opt(1).T);


% lsqnonlin


%% 3: Portfolio Optimisation, Track A
clear;
ASSETS=readtable('ASSETSA.csv');


Date=ASSETS.Date;
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
for i=1:12
subplot(4,3,i)
plot(Date,PRICES(:,i))
title(FULLNAMES{i})
end
figure(2)
for i=1:12
subplot(4,3,i)
plot(Date(2:end),LOGRETURNS(:,i))
title(FULLNAMES{i})
end
