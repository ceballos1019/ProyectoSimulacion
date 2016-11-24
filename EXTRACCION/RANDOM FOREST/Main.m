
clc
clear all
% close all

load('datosPhishing.mat');  %%Cargar los datos
load('matrizTransformacion.mat'); %%Cargar la matriz de trasnformacion para hacer la extraccion
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra

%Reevaluar el modelo usando extraccion de caracteristicas
X = X * matrizTransform;

NumMuestras=size(X,1); %%Numero de filas => numero de muestras
Rept=10;  %%Repeticiones
eficiencia=zeros(9,Rept); %%vector fila con "rept" elementos
sensibilidad=zeros(9,Rept);
especificidad=zeros(9,Rept);
precision=zeros(9,Rept);
NumArboles=[5,10,20,30,40,50,100,200,500];

    
%%% punto Random Forest %%%
    
NumClases=length(unique(Y)); %%% Se determina el n�mero de clases del problema.
%NumArboles = input('Ingrese el numero de arboles: ');
%tic;
for n=1:9
    for fold=1:Rept

        %%% Se hace la partici�n de las muestras %%%
        %%%      de entrenamiento y prueba       %%%

        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept); %%Validacion cruzada, k sera Rept, K subconjuntos de "igual" tama�o
        indices=particion.training(fold); %%Retorna un vector logico que indica que muestras son para entrenar (training) y cuales para validar (test)
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Normalizaci�n %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

       Modelo=entrenarFOREST(NumArboles(n),Xtrain,Ytrain');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Validaci�n de los modelos. %%%

       Yesti=testFOREST(Modelo,Xtest);
       FN=sum(Yesti==-1 & Yesti~=Ytest);
       FP=sum(Yesti==1 & Yesti~=Ytest);
       TP=sum(Yesti==Ytest & Yesti==1);
       TN=sum(Yesti==Ytest)-TP;
       sensibilidad(n,fold)=(TP)/(TP+FN);
       especificidad(n,fold)=(TN)/(TN+FP);
       precision(n,fold)=(TP)/(TP+FP);
       eficiencia(n,fold)=(TP+TN)/(TP+TN+FP+FN); 
       texto=['Arboles = ', num2str(NumArboles(n)),' fold: ',num2str(fold)];
       disp(texto);        
    end   
end
eficienciaFinalEXT_RF=zeros(9,2);
especificidadFinalEXT_RF=zeros(9,2);
sensibilidadFinalEXT_RF=zeros(9,2);
precisionFinalEXT_RF=zeros(9,2);
for i=1:9
    eficienciaFinalEXT_RF(i,1)=mean(eficiencia(i,:));
    eficienciaFinalEXT_RF(i,2)=std(eficiencia(i,:));
    especificidadFinalEXT_RF(i,1)=mean(especificidad(i,:));
    especificidadFinalEXT_RF(i,2)=std(especificidad(i,:));
    sensibilidadFinalEXT_RF(i,1)=mean(sensibilidad(i,:));
    sensibilidadFinalEXT_RF(i,2)=std(sensibilidad(i,:));
    precisionFinalEXT_RF(i,1)=mean(precision(i,:));
    precisionFinalEXT_RF(i,2)=std(precision(i,:));
end
save('eficienciaFinalEXT_RF.mat','eficienciaFinalEXT_RF');
save('especificidadFinalEXT_RF.mat','especificidadFinalEXT_RF');
save('sensibilidadFinalrfEXT_RF.mat','sensibilidadFinalEXT_RF');
save('precisionFinalrfEXT_RF.mat','precisionFinalEXT_RF');





