clc
clear all
close all


load('datosPhishing.mat');  %%Cargar los datos
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra


Rept=10;
NumMuestras=size(X,1);
umbralPorcentajeDeVarianza = 90;
%[Xtrain,mu,sigma] = zscore(Xtrain);
[coefCompPrincipales,scores,covarianzaEigenValores,~,porcentajeVarianzaExplicada,~] = pca(X);
numVariables = length(covarianzaEigenValores);
numCompAdmitidos = 0;
porcentajeVarianzaAcumulada = zeros(numVariables,1);
puntosUmbral = ones(numVariables,1)*umbralPorcentajeDeVarianza;

for k=1:numVariables
    porcentajeVarianzaAcumulada(k) = sum(porcentajeVarianzaExplicada(1:k));

    if (sum(porcentajeVarianzaExplicada(1:k)) >= umbralPorcentajeDeVarianza) && (numCompAdmitidos == 0)
        numCompAdmitidos = k;
    end
end

matrizTransform = coefCompPrincipales(:,1:numCompAdmitidos);

save('matrizTransformacion.mat','matrizTransform');

texto=['Porcentaje de varianza: ',num2str(umbralPorcentajeDeVarianza),' - Logrado con ', num2str(numCompAdmitidos),' componentes principales'];
disp(texto);

%{
aux = Xtest*coefCompPrincipales;
Xtest = aux(:,1:numCompAdmitidos);
%}
    

