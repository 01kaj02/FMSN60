function rho=sacfplot(yin,n)

% Ex. Ange sacfplot(yin,20)
% där 20 anger \rho(0)..\rho(20)

% Copyright Erik Lindström, 020315
     
R=covf(yin,n+1);
rho=R/R(1);

x=ones(n+1,1);
figure
plot(0:n,rho,0:n,rho,'.')
hold on
plot(0:n,1.96/sqrt(length(yin))*x,'-.');
plot(0:n,1.96/sqrt(length(yin))*x*-1,'-.');
grid
xlabel('lag')