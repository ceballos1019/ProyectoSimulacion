function Modelo = entrenarRNAClassication(X,Y,NumeroNeuronas)

%%% Completar el codigo %%%
%trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation. -- default
%trainFcn = 'trainlm';  %Levenberg-Marquardt backpropagation

net = patternnet(NumeroNeuronas);

 net.divideFcn = 'dividerand';  % Divide data randomly
 net.divideMode = 'sample';  % Divide up every sample
 net.divideParam.trainRatio = 70/100;
 net.divideParam.valRatio = 30/100;
 
 % Choose a Performance Function
 %net.performFcn = 'crossentropy';  % Cross-Entropy - default
 
 Modelo = train(net,X',Y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
