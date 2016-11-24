clear;
clc;

load('datosPhishing');
X=datosPhishing(:,1:30);
Y=datosPhishing(:,end);

cvp = cvpartition(Y, 'k', 10);
tic;
features = sequentialfs(@functionForest, X, Y, 'cv', cvp);
toc;


