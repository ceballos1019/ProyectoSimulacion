function modelo = entrenarGMM(X,NumeroMezclas)    
    inputDim = size(X,2);      %%%%% Numero de caracteristicas de las muestras
    mezclas = gmm(inputDim, NumeroMezclas, 'spherical'); %%crea el modelo con dimension=inputDim, numero de centros=numeroMezclas y la estructura de la matriz de covarianza ('spherical','diag','full')
    options = foptions;
    options(14)=10; %%numero de iteraciones del k-means 
    mezclas = gmminit(mezclas, X, options); %%Inicaliza el modelo con las muestras de X, usa el algoritmo k means para hallar los centros
    modelo = gmmem(mezclas, X, options); %%Usa algoritmo de esperanza y maximizacion par estimar los parametros del modelo gaussiano, options es el numero de iteraciones (100 por defecto)
    
end


%{
Estructura del modelo:
  type = 'gmm'
  nin = the dimension of the space
  ncentres = number of mixture components
  covartype = string for type of variance model
  priors = mixing coefficients
  centres = means of Gaussians: stored as rows of a matrix
  covars = covariances of Gaussians
%}
