clc;
clear all;
close all;

[x,labels] = generateDataA2(10000);



function [x,labels] = generateDataA2(N)
%N = 100;
figure(1), clf,     %colors = 'bm'; markers = 'o+';
classPriors = [0.3,0.3,0.4];
labels = (rand(1,N) >= classPriors(3));
for l = 0:1
    indl = find(labels==l);
    if l == 0
        N0 = length(indl);
        mu0 = [3;0;0];
        Sigma0 = [2 0 0;0 1 0;0 0 1];
        x(:,indl) = mvnrnd(mu0,Sigma0,N0)';
        plot3(x(1,indl),x(2,indl),x(3,indl),'ro'), hold on,
        grid on;
        title('10000 Samples Generated','Interpreter','latex');
        xlabel('$x_1$','Interpreter','latex');
        ylabel('$x_2$','Interpreter','latex');
        zlabel('$x_3$','Interpreter','latex');

    elseif l == 1
        N1 = length(indl);
        mu1 = [0;3;0]; 
        N2 = length(indl);
        mu2 = [0;0;3];
        Sigma1 = [1 0 0; 0 2 0; 0 0 1];
        Sigma2 = [1 0 0; 0 1 0; 0 0 2];
        x(:,indl) = mvnrnd(mu1,Sigma1,N1)';
        plot3(x(1,indl),x(2,indl),x(3,indl),'b+'), hold on,
        x(:,indl) = mvnrnd(mu2,Sigma2,N2)';
        plot3(x(1,indl),x(2,indl),x(3,indl),'gx'), hold on,
        axis equal,
    end
    legend('$p(\mathbf{x}|L=1)$','$p(\mathbf{x}|L=2)$','$p(\mathbf{x}|L=3)$','Interpreter','latex');
end
end