function matrix = normalize(matrix, mean, std)
    N = size(matrix, 1);            % Obteniendo el tamaño de la matriz.

    % Aplicando la formulación de Normalización por ZSCORE.
    matrix = (matrix - ones(N, 1) * mean) ./ (ones(N, 1) * std);