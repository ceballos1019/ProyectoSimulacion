    
clc
clear all
close all

load('datosPhishing.mat');  %%Cargar los datos

X=datosPhishing(:,1:30);  %%Toma las 30 columnas de los datos que corresponden a las muestras
Y=datosPhishing(:,end);  %%Toma la ultima columna que corresponde a las clases de cada muestra
NumClases=length(unique(Y)); %%% Se determina el numero de clases del problema.
NumMuestras=size(X,1);
Rept=10;
sensibilidad=zeros(1,Rept);
especificidad=zeros(1,Rept);
precision=zeros(1,Rept);
eficiencia=zeros(1,Rept);
NumMezclas = 1;

for fold=1:Rept
    %%% Se hace la partici�n de las muestras %%%
    %%%      de entrenamiento y prueba       %%%
        
    rng('default');
    particion=cvpartition(NumMuestras,'Kfold',Rept); %%Validacion cruzada, k sera Rept, K subconjuntos de "igual" tama�o
    indices=particion.training(fold);  %%Retorna un vector logico que indica que muestras son para entrenar (training) y cuales para validar (test)
    
    %%Se construyen los conjuntos de entrenamiento y de prueba con los
    %%indices
    
    Xtrain=X(particion.training(fold),:);
    Xtest=X(particion.test(fold),:);
    Ytrain=Y(particion.training(fold));
    Ytest=Y(particion.test(fold));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se normalizan los datos %%%

    [Xtrain,mu,sigma] = zscore(Xtrain);
    %%testing=repmat(mu,size(Xtest,1),1);
    Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1); %%(Xtest - media)/std
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

    vInd=(Ytrain == 1);  %%Obtiene los indices en la salida Y que corresponden a la clase 1
    XtrainC1 = Xtrain(vInd,:); %%Obtiene las muestras de entrenamiento de la clase 1
    if ~isempty(XtrainC1)
        Modelo1=entrenarGMM(XtrainC1,NumMezclas);
    else
        error('No hay muestras de todas las clases para el entrenamiento');
    end
        
    vInd=(Ytrain == -1); %%Obtiene los indices en la salida Y que corresponden a la clase 2
    XtrainC2 = Xtrain(vInd,:); %%Obtiene las muestras de entrenamiento de la clase 2
    if ~isempty(XtrainC2)
        Modelo2=entrenarGMM(XtrainC2,NumMezclas);
    else
        error('No hay muestras de todas las clases para el entrenamiento');
    end        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %%% Validaci�n de los modelos. %%%
    %Calcula la probabilidad del conjunto de validacion en cada clase%
    probClase1=testGMM(Modelo1,Xtest);
    probClase2=testGMM(Modelo2,Xtest);        
    Matriz=[probClase1,probClase2]; %%Matriz con 2 columnas, cada una es la probabilidad en cada clase de las muestras de validacion
        
    [~,Yest] = max(Matriz,[],2); %%Maximo valor de cada fila --> la clase a la que cada muestra tuvo mayor probabilidad
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %%Se crea la matriz de confusion%%
    MatrizConfusion = zeros(NumClases,NumClases);
        
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
    sensibilidad(fold)=(TP)/(TP+FN);
    especificidad(fold)=(TN)/(TN+FP);
    precision(fold)=(TP)/(TP+FP);
    eficiencia(fold)=(TP+TN)/(TP+TN+FP+FN);        
end
eficienciaFinalFDG = zeros(1,2);
sensibilidadFinalFDG = zeros(1,2);
especifiidadFinalFDG = zeros(1,2);
precisionFinalFDG = zeros(1,2);

eficienciaFinalFDG(1,1)=mean(eficiencia(1,:));
eficienciaFinalFDG(1,2)=std(eficiencia(1,:));
especificidadFinalFDG(1,1)=mean(especificidad(1,:));
especificidadFinalFDG(1,2)=std(especificidad(1,:));
sensibilidadFinalFDG(1,1)=mean(sensibilidad(1,:));
sensibilidadFinalFDG(1,2)=std(sensibilidad(1,:));
precisionFinalFDG(1,1)=mean(precision(1,:));
precisionFinalFDG(1,2)=std(precision(1,:));

save('eficienciaFinalFDG.mat','eficienciaFinalFDG');
save('sensibilidadFinalFDG.mat','sensibilidadFinalFDG');
save('especificidadFinalFDG.mat','especificidadFinalFDG');
save('precisionFinalFDG.mat','precisionFinalFDG');