function Modelo = entrenarFOREST(NumArboles,X,Y)

   	Modelo = TreeBagger(NumArboles,X,Y);

end


%{
	Bag of decision trees
	By default, TreeBagger bags classification trees. To bag regression trees instead, specify 'Method','regression'.
	B = TreeBagger(NumTrees,X,Y) 
	creates an ensemble B of NumTrees decision trees for predicting response Y as a function of predictors in the numeric matrix of training data, X. 
	Each row in X represents an observation and each column represents a predictor or feature.

	B = TreeBagger(NumTrees,X,Y,Name,Value) specifies optional parameter name-value pairs:
	'NumPredictorsToSample' → Number of variables to select at random for each decision split.
	Default is the square root of the number of variables for classification and one third of the number of variables for regression. 
	Valid values are 'all' or a positive integer. 


%}