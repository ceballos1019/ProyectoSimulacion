clear;
clc;


load('datosPhishing.mat');
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra

dataForClasses = classificatedata(X, Y);

fishers = calculatefisher(dataForClasses);

posibleEvaluar1 = fishers(fishers<0.01);
posibleEvaluar2 = fishers(fishers<0.001);