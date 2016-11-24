
clc
clear all
close all

load('datosPhishing.mat');  %%Cargar los datos
X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra
NumMuestras=size(X,1); %%Numero de filas => numero de muestras
NumClases=length(unique(Y)); %%% Se determina el n?mero de clases del problema.
Rept=10;  %%Repeticiones
eficiencia=zeros(1,Rept); %%vector fila con "rept" elementos
sensibilidad=zeros(1,Rept);
especificidad=zeros(1,Rept);
precision=zeros(1,Rept);
NumNeuronas=[10,20,30,40,50,100,150];
redesY = zeros(NumMuestras,NumClases);

for n=1:7
    for i=1:NumMuestras
        if(Y(i)==1)
            redesY(i,1)=1;
        else
            redesY(i,2)=1;
        end
    end
    for fold=1:Rept

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%

        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        %Ytrain=Y(particion.training(fold),:);
        Ytrain = redesY(particion.training(fold),:);
        %[~,Ytest]=max(Y(particion.test(fold),:),[],2);
        [~,Ytest]=max(redesY(particion.test(fold),:),[],2);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [XtrainNormal,mu,sigma]=zscore(Xtrain);
        XtestNormal=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        Modelo=entrenarRNAClassication(Xtrain,Ytrain,NumNeuronas(n));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Validación de los modelos. %%%

        Yesti=testRNA(Modelo,Xtest);
        [~,Yesti]=max(Yesti,[],2);

        FN=sum(Yesti==2 & Yesti~=Ytest);
        FP=sum(Yesti==1 & Yesti~=Ytest);
        TP=sum(Yesti==Ytest & Yesti==1);
        TN=sum(Yesti==Ytest)-TP;
        sensibilidad(n,fold)=(TP)/(TP+FN);
        especificidad(n,fold)=(TN)/(TN+FP);
        precision(n,fold)=(TP)/(TP+FP);
        eficiencia(n,fold)=(TP+TN)/(TP+TN+FP+FN); 
        %{
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)       
            MatrizConfusion(Yesti(i),Ytest(i)) = MatrizConfusion(Yesti(i),Ytest(i)) + 1;
        end
        eficiencia(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        %}
        texto=['Neuronas = ', num2str(NumNeuronas(n)),' fold: ',num2str(fold)];
        disp(texto);
    end   
end
eficienciaFinalRNA=zeros(7,2);
especificidadFinalRNA=zeros(7,2);
sensibilidadFinalRNA=zeros(7,2);
precisionFinalRNA=zeros(7,2);
for i=1:7
    eficienciaFinalRNA(i,1)=mean(eficiencia(i,:));
    eficienciaFinalRNA(i,2)=std(eficiencia(i,:));
    especificidadFinalRNA(i,1)=mean(especificidad(i,:));
    especificidadFinalRNA(i,2)=std(especificidad(i,:));
    sensibilidadFinalRNA(i,1)=mean(sensibilidad(i,:));
    sensibilidadFinalRNA(i,2)=std(sensibilidad(i,:));
    precisionFinalRNA(i,1)=mean(precision(i,:));
    precisionFinalRNA(i,2)=std(precision(i,:));
end
save('eficienciaFinalRNA.mat','eficienciaFinalRNA');
save('especificidadFinalRNA.mat','especificidadFinalRNA');
save('sensibilidadFinalRNA.mat','sensibilidadFinalRNA');
save('precisionFinalRNA.mat','precisionFinalRNA');




