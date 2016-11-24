clc
clear all
close all

load('datosPhishing.mat');
X=datosPhishing(:,1:30);
Y=datosPhishing(:,end);

%Reevaluar el modelo usando solo las caracteristicas seleccionadas

X1=X(:,1:2);
X2=X(:,6:10);
X3=X(:,12:15);
X4=X(:,17);
X5=X(:,24:30);
X=[X1,X2,X3,X4,X5];

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
eficienciaFinalSEL_KN=zeros(7,2);
especificidadFinalSEL_KN=zeros(7,2);
sensibilidadFinalSEL_KN=zeros(7,2);
precisionFinalSEL_KN=zeros(7,2);
for i=1:7
    eficienciaFinalSEL_KN(i,1)=mean(eficiencia(i,:));
    eficienciaFinalSEL_KN(i,2)=std(eficiencia(i,:));
    especificidadFinalSEL_KN(i,1)=mean(especificidad(i,:));
    especificidadFinalSEL_KN(i,2)=std(especificidad(i,:));
    sensibilidadFinalSEL_KN(i,1)=mean(sensibilidad(i,:));
    sensibilidadFinalSEL_KN(i,2)=std(sensibilidad(i,:));
    precisionFinalSEL_KN(i,1)=mean(precision(i,:));
    precisionFinalSEL_KN(i,2)=std(precision(i,:));
end
save('eficienciaFinalSEL_KN.mat','eficienciaFinalSEL_KN');
save('especificidadFinalSEL_KN.mat','especificidadFinalSEL_KN');
save('sensibilidadFinalSEL_KN.mat','sensibilidadFinalSEL_KN');
save('precisionFinalknSEL_KN.mat','precisionFinalSEL_KN');