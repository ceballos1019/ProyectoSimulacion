function value = classificatedata(X, Y)    
    indexesA = (Y == 1);
    value{1} = X(indexesA,:);
    indexesB = (Y == -1);
    value{2} = X(indexesB,:);

    