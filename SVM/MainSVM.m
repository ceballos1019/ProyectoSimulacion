clc;
clear all;
close all;

load('datosPhishing.mat');
X=datosPhishing(:,1:30);
Y=datosPhishing(:,end);
NumClases=length(unique(Y)); %%% Se determina el numero de clases del problema.
NumMuestras=size(X,1);
Rept=10;
sensibilidad=zeros(5,Rept);
especificidad=zeros(5,Rept);
precision=zeros(5,Rept);
eficiencia=zeros(5,Rept);
gamma=[0.01 0.1 1 10 100];
boxConstraint=[0.01 0.1 1 10 100];
%tic;
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
    eficienciaFinalsvm=zeros(5,2);
    especificidadFinalsvm=zeros(5,2);
    sensibilidadFinalsvm=zeros(5,2);
    precisionFinalsvm=zeros(5,2);
    for i=1:5
        eficienciaFinalsvm(i,1)=mean(eficiencia(i,:));
        eficienciaFinalsvm(i,2)=std(eficiencia(i,:));
        especificidadFinalsvm(i,1)=mean(especificidad(i,:));
        especificidadFinalsvm(i,2)=std(especificidad(i,:));
        sensibilidadFinalsvm(i,1)=mean(sensibilidad(i,:));
        sensibilidadFinalsvm(i,2)=std(sensibilidad(i,:));
        precisionFinalsvm(i,1)=mean(precision(i,:));
        precisionFinalsvm(i,2)=std(precision(i,:));
    end    
    texto1=['eficienciaFinalsvm',num2str(boxind),'.mat'];
    texto2=['especificidadFinalsvm',num2str(boxind),'.mat'];
    texto3=['sensibilidadFinalsvm',num2str(boxind),'.mat'];
    texto4=['precisionFinalsvm',num2str(boxind),'.mat'];
    save(texto1,'eficienciaFinalsvm');
    save(texto2,'especificidadFinalsvm');
    save(texto3,'sensibilidadFinalsvm');
    save(texto4,'precisionFinalsvm');
 end
%toc;