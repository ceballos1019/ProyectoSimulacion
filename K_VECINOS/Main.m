clc
clear all
close all

load('datosPhishing.mat');
X=datosPhishing(:,1:30);
Y=datosPhishing(:,end);

%%% Se hace la particiï¿½n entre los conjuntos de entrenamiento y prueba.
%%% Esta particiï¿½n se hace forma aletoria %%%
    
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
eficienciaFinalkn=zeros(7,2);
especificidadFinalkn=zeros(7,2);
sensibilidadFinalkn=zeros(7,2);
precisionFinalkn=zeros(7,2);
for i=1:7
    eficienciaFinalkn(i,1)=mean(eficiencia(i,:));
    eficienciaFinalkn(i,2)=std(eficiencia(i,:));
    especificidadFinalkn(i,1)=mean(especificidad(i,:));
    especificidadFinalkn(i,2)=std(especificidad(i,:));
    sensibilidadFinalkn(i,1)=mean(sensibilidad(i,:));
    sensibilidadFinalkn(i,2)=std(sensibilidad(i,:));
    precisionFinalkn(i,1)=mean(precision(i,:));
    precisionFinalkn(i,2)=std(precision(i,:));
end
save('eficienciaFinalkn.mat','eficienciaFinalkn');
save('especificidadFinalkn.mat','especificidadFinalkn');
save('sensibilidadFinalkn.mat','sensibilidadFinalkn');
save('precisionFinalkn.mat','precisionFinalkn');