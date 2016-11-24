function Error = functionForest(Xtrain, Ytrain,Xtest, Ytest)

    %Normalizar los datos
    [Xtrain, means, stds] = zscore(Xtrain);
    Xtest = normalize(Xtest, means, stds);
    
    %Entrenar el modelo
    NumClases=length(unique(Ytrain));
    NumArboles=50;
    Modelo = TreeBagger(NumArboles,Xtrain,Ytrain');
    %disp(NumClases);
    
    %Probar el modelo
    Yest = predict(Modelo,Xtest);
    Yest = str2double(Yest);  
    
    %Calcular error
    MatrizConfusion = zeros(NumClases,NumClases);
    %disp(size(Xtest,1));
    for i=1:size(Xtest,1)
        posTest= 1;
            posEst = 1;            
            if Yest(i) == -1
                posEst = 2;
            end            
            if Ytest(i)== -1
                posTest = 2;
            end
            MatrizConfusion(posEst,posTest) = MatrizConfusion(posEst,posTest) + 1;
    end
    Eficiencia = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    
    Error = 1 - Eficiencia;