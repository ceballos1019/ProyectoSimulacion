clc;
clear all;
close all;

load('datosPhishing.mat');  %%Cargar los datos
load('matrizTransformacion.mat'); %%Cargar la matriz de trasnformacion para hacer la extraccion
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra

%Reevaluar el modelo usando extraccion de caracteristicas
X = X * matrizTransform;

NumClases=length(unique(Y)); %%% Se determina el numero de clases del problema.
NumMuestras=size(X,1);
Rept=10;
sensibilidad=zeros(5,Rept);
especificidad=zeros(5,Rept);
precision=zeros(5,Rept);
eficiencia=zeros(5,Rept);
gamma=[0.01 0.1 1 10 100];
boxConstraint=[0.01 0.1 1 10 100];
tic;
 for boxind=1:5
    for gammaind=1:5        
        for fold=1:Rept
            %%% Se hace la partici?n entre los conjuntos de entrenamiento y prueba.
            %%% Esta partici?n se hace forma aletoria %%%
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold));
            Ytest=Y(particion.test(fold));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Se normalizan los datos %%%

            [Xtrain,mu,sigma]=zscore(Xtrain);
            Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Complete el codigo implimentando la estrategia One vs All.
            %%% Recuerde que debe de entrenar un modelo SVM para cada clase.
            %%% Solo debe de evaluar las muestras con conflicto.

            Ytrain1 = Ytrain;
            Ytrain1(Ytrain ~= 1) = -1;
            Modelo1=entrenarSVM(Xtrain,Ytrain1,'classification',boxConstraint(boxind),gamma(gammaind));

            Ytrain2 = Ytrain;
            Ytrain2(Ytrain ~= -1) = -1;
            Ytrain2(Ytrain == -1) = 1;
            Modelo2=entrenarSVM(Xtrain,Ytrain2,'classification',boxConstraint(boxind),gamma(gammaind));            

            [~,Yest1]=testSVM(Modelo1,Xtest);
            [~,Yest2]=testSVM(Modelo2,Xtest);            

            [~,Yest] = max([Yest1,Yest2],[],2); 

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            MatrizConfusion=zeros(NumClases,NumClases);
            for i=1:size(Xtest,1)
                posTest= 1;               
                if Ytest(i)== -1
                    posTest = 2;
                end
                MatrizConfusion(Yest(i),posTest) = MatrizConfusion(Yest(i),posTest) + 1;
            end
            TP=MatrizConfusion(1,1);
            TN=MatrizConfusion(2,2);
            FN=MatrizConfusion(2,1);
            FP=MatrizConfusion(1,2);
            sensibilidad(gammaind,fold)=(TP)/(TP+FN);
            especificidad(gammaind,fold)=(TN)/(TN+FP);
            precision(gammaind,fold)=(TP)/(TP+FP);
            eficiencia(gammaind,fold)=(TP+TN)/(TP+TN+FP+FN);
            texto=['Gamma = ', num2str(gamma(gammaind)),' fold: ',num2str(fold), ' Box: ',num2str(boxConstraint(boxind))];
            disp(texto);
        end 
    end
    eficienciaFinalEXT_SVM=zeros(5,2);
    especificidadFinalEXT_SVM=zeros(5,2);
    sensibilidadFinalEXT_SVM=zeros(5,2);
    precisionFinalEXT_SVM=zeros(5,2);
    for i=1:5
        eficienciaFinalEXT_SVM(i,1)=mean(eficiencia(i,:));
        eficienciaFinalEXT_SVM(i,2)=std(eficiencia(i,:));
        especificidadFinalEXT_SVM(i,1)=mean(especificidad(i,:));
        especificidadFinalEXT_SVM(i,2)=std(especificidad(i,:));
        sensibilidadFinalEXT_SVM(i,1)=mean(sensibilidad(i,:));
        sensibilidadFinalEXT_SVM(i,2)=std(sensibilidad(i,:));
        precisionFinalEXT_SVM(i,1)=mean(precision(i,:));
        precisionFinalEXT_SVM(i,2)=std(precision(i,:));
    end    
    texto1=['eficienciaFinalEXT_SVM',num2str(boxind),'.mat'];
    texto2=['especificidadFinalEXT_SVM',num2str(boxind),'.mat'];
    texto3=['sensibilidadFinalEXT_SVM',num2str(boxind),'.mat'];
    texto4=['precisionFinalEXT_SVM',num2str(boxind),'.mat'];
    save(texto1,'eficienciaFinalEXT_SVM');
    save(texto2,'especificidadFinalEXT_SVM');
    save(texto3,'sensibilidadFinalEXT_SVM');
    save(texto4,'precisionFinalEXT_SVM');
 end
toc;