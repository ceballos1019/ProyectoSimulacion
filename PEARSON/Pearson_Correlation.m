clear;
clc;

load('datosPhishing.mat');
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra

correlationMatrixWithFeatures = corrcoef(X);
save('CorrelationMatrixWithFeatures.mat', 'correlationMatrixWithFeatures');

temp = [X, Y];
correlationMatrixWithOutput = corrcoef(temp);
save('CorrelationMatrixWithOutput.mat', 'correlationMatrixWithOutput');

clear;
load('CorrelationMatrixWithFeatures.mat');
load('CorrelationMatrixWithOutput.mat');