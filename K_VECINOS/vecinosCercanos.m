function Yesti = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de modelo de que se va a entrenar

    N=size(Xent,1);
    M=size(Xval,1);
    
    Yesti=zeros(M,1);
    dis=zeros(N,1);

    if strcmp(tipo,'class')
        
        for j=1:M
            %%% Complete el codigo %%%
             dis=distancia(Xent, Xval(j,:));
                        
            [dis, sortIndexes]= sort(dis);
            
            temp = Yent(sortIndexes(1:k));
            Yesti(j)= mode (temp);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        
    elseif strcmp(tipo,'regress')
        
        for j=1:M
            %%% Complete el codigo %%%
			dis=sqrt(Xval-Xent)
			%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end

        
    end

end