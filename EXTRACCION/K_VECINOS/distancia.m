function distances = distancia(Xtrain, Xval)
    N = size(Xtrain, 1); %%Numero de muestras totales de entrenamiento
    distances = zeros(N, 1); %%Vector columna de N elementos lleno de ceros

    for i = 1:N
        temp = (Xtrain(i, :) - Xval) .^ 2;  %% Muestra entrenamiento i - Muestra validacion
        distances(i) = sqrt(sum(temp, 2));
    end