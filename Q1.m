clc;
clear all;
close all;
classPriors = [0.65,0.35];
w0 = [0.5,0.5];
mu(:,1) = [3;0];
mu(:,2) = [0;3];
mu(:,3) = [2;2];
Sigma(:,:,1) = [2,0;0,1];
Sigma(:,:,2) = [1,0;0,2];
Sigma(:,:,3) = [1,0;0,1];
[x,labels] = generateDataA1(10000);
[FPR,TPR] = plotROC(x,labels,w0,mu,Sigma);


function [x,labels] = generateDataA1(N)
%N = 100;
figure(1), clf,     %colors = 'bm'; markers = 'o+';
classPriors = [0.65,0.35];
labels = (rand(1,N) >= classPriors(1));
for l = 0:1
    indl = find(labels==l);
    if l == 0
        N0 = length(indl);
        w0 = [0.5,0.5]; mu0 = [3 0;0 3];
        Sigma0(:,:,1) = [2 0;0 1]; Sigma0(:,:,2) = [1 0;0 2];
        gmmParameters.priors = w0; % priors should be a row vector
        gmmParameters.meanVectors = mu0;
        gmmParameters.covMatrices = Sigma0;
        [x(:,indl),components] = generateDataFromGMM(N0,gmmParameters);
        %plot(x(1,indl(components==1)),x(2,indl(components==1)),'ro'), hold on,
        %plot(x(1,indl(components==2)),x(2,indl(components==2)),'ro'), hold on,
        plot(x(1,indl),x(2,indl),'ro'), hold on,
        title('10000 Samples Generated','Interpreter','latex');
        xlabel('$x_1$','Interpreter','latex');
        ylabel('$x_2$','Interpreter','latex');

    elseif l == 1
        m1 = [2;2]; C1 = eye(2);
        N1 = length(indl);
        x(:,indl) = mvnrnd(m1,C1,N1)';
        plot(x(1,indl),x(2,indl),'b+'), hold on,
        axis equal,
    end
    legend('$p(\mathbf{x}|L=0)$','$p(\mathbf{x}|L=1)$','Interpreter','latex');
end
end

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l));
    Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end

function [FPR,TPR,TNR] = plotROC(x,labels,w0,mu,Sigma)
count = [length(find(labels == 0)),length(find(labels == 1))];
log_g1 = log(G(x,mu(:,3),Sigma(:,:,3)));
log_g2 = log(w0(1)*(G(x,mu(:,1),Sigma(:,:,1))) + w0(2)*(G(x,mu(:,2),Sigma(:,:,2))));
dScore = log_g1-log_g2;
t = log(sort(dScore(dScore>=0)));
mid_t = [t(1)-1,t(1:length(t)-1) + diff(t)./2,t(length(t))+1];
for i = 1:length(mid_t)
    d = (dScore >= mid_t(i));
    FPR(i) = sum(d==1 & labels==0)/count(1);
    TPR(i) = sum(d==1 & labels==1)/count(2);
    TNR(i) = sum(d==0 & labels==1)/count(2);
    ER(i) = FPR(i)*0.65+(1-TPR(i))*0.35;
    ER_E(i) = FPR(i)*0.65+TNR(i)*0.35;
end
[min_error, min_index] = min(ER);
[min_error2,min_index2] = min(ER_E);
min_FP = FPR(min_index);
min_TP = TPR(min_index);

min_FP2 = FPR(min_index2);
min_TP2 = TPR(min_index2);
disp(min_FP);
disp(min_TP);
disp(min_FP2);
disp(min_TP2);

figure(2);
plot(FPR,TPR,'-',min_FP,min_TP,'o','MarkerFaceColor','r');
grid on;
hold on;
plot(min_FP2,min_TP2,'bx');
title('ROC Curve of the Minimum Expected Risk Classifier','Interpreter','latex');
xlabel('False Positive Rate $P(D=1|L=0)$','Interpreter','latex'); 
ylabel('True Positive Rate $P(D=1|L=1)$','Interpreter','latex');
legend('ROC Curve', 'Theoretical threshold value','Emprical threshold value','Interpreter','latex');
end

function G_pdf = G(x,mu,Sigma)
[n,N] = size(x);
term1 = (det(Sigma)*(2*pi)^n)^(-1/2);
term2 = (-1/2)*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
G_pdf = term1*exp(term2);
end