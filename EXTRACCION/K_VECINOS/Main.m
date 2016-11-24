clc
clear all
close all

load('datosPhishing.mat');  %%Cargar los datos
load('matrizTransformacion.mat'); %%Cargar la matriz de trasnformacion para hacer la extraccion
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra

%Reevaluar el modelo usando extraccion de caracteristicas
X = X * matrizTransform;

N=size(X,1); %%Numero de muestras
Rept=10;  %%Repeticiones
eficiencia=zeros(7,Rept); %%vector fila con "rept" elementos
sensibilidad=zeros(7,Rept);
especificidad=zeros(7,Rept);
precision=zeros(7,Rept);
NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.    
vecinos=[1,3,5,7,9,11,13];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%k=1;
%k = input('Ingrese el numero de vecinos: ');
%tic;
for k=1:7
    for fold=1:Rept
    rng('default');
    particion=cvpartition(N,'Kfold',Rept); %%Validacion cruzada, k sera Rept, K subconjuntos de "igual" tamaño
    indices=particion.training(fold);  %%Retorna un vector logico que indica que muestras son para entrenar (training) y cuales para validar (test)

    %%Se construyen los conjuntos de entrenamiento y de prueba con los
    %%indices
       
    Xtrain=X(particion.training(fold),:);
    Xtest=X(particion.test(fold),:);
    Ytrain=Y(particion.training(fold));
    Ytest=Y(particion.test(fold));
    
    %%% Normalizaciï¿½n %%%
    
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%
    
     %%% Se aplica la clasificaciï¿½n con KNN %%%
    
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se encuentra la eficiencia y el error de clasificaciï¿½n %%%
    Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,vecinos(k),'class');
    FN=sum(Yesti==-1 & Yesti~=Ytest);
    FP=sum(Yesti==1 & Yesti~=Ytest);
    TP=sum(Yesti==Ytest & Yesti==1);
    TN=sum(Yesti==Ytest)-TP;
    sensibilidad(k,fold)=(TP)/(TP+FN);
    especificidad(k,fold)=(TN)/(TN+FP);
    precision(k,fold)=(TP)/(TP+FP);
    eficiencia(k,fold)=(TP+TN)/(TP+TN+FP+FN); 
    texto=['vecinos = ', num2str(vecinos(k)),' fold: ',num2str(fold)];
    disp(texto);        
    end    
end
eficienciaFinalEXT_KN=zeros(7,2);
especificidadFinalEXT_KN=zeros(7,2);
sensibilidadFinalEXT_KN=zeros(7,2);
precisionFinalEXT_KN=zeros(7,2);
for i=1:7
    eficienciaFinalEXT_KN(i,1)=mean(eficiencia(i,:));
    eficienciaFinalEXT_KN(i,2)=std(eficiencia(i,:));
    especificidadFinalEXT_KN(i,1)=mean(especificidad(i,:));
    especificidadFinalEXT_KN(i,2)=std(especificidad(i,:));
    sensibilidadFinalEXT_KN(i,1)=mean(sensibilidad(i,:));
    sensibilidadFinalEXT_KN(i,2)=std(sensibilidad(i,:));
    precisionFinalEXT_KN(i,1)=mean(precision(i,:));
    precisionFinalEXT_KN(i,2)=std(precision(i,:));
end
save('eficienciaFinalEXT_KN.mat','eficienciaFinalEXT_KN');
save('especificidadFinalEXT_KN.mat','especificidadFinalEXT_KN');
save('sensibilidadFinalEXT_KN.mat','sensibilidadFinalEXT_KN');
save('precisionFinalknEXT_KN.mat','precisionFinalEXT_KN');