clear;  % ***Clear all previous data before executing this code***

% prepare data
table = readtable('adult.csv');
attribute_name = table.Properties.VariableNames;
table.workclass = double (categorical(table.workclass));
table.fnlwgt = normalize (table.fnlwgt,'range'); %data scaling, [0 1]
table.marital_status = double(categorical(table.marital_status));
table.occupation = double(categorical(table.occupation));
table.relationship = double(categorical(table.relationship));
table.education = double(categorical(table.education));
table.race = double(categorical(table.race));
table.sex = double(categorical(table.relationship));
table.native_country= double(categorical(table.native_country));
table.census_income = double(categorical(table.census_income));
table.census_income = table.census_income - 1;



% cross validation initial variables
k = 10;
drop_samples=mod(size(table,1),k);
table = table(1:size(table,1)-drop_samples,:);
%implement index scrambling here
sub_sample=table (1:5000,:);

dat_convert=table2array(sub_sample);
inputs = dat_convert (:,1:14);
targets = dat_convert (:, 15);

numOfExamples = (k-1)*size(inputs,1)/k; 
numOfFeatures = size(inputs,2);         
foldLength = size(inputs,1)/k;          
P = randperm(size(inputs,1));         

%Outerloop return variables
recalls = zeros(1,10);
precisions = zeros(1,10);
accuracy = zeros(1,10);
best_C_values = zeros(1,10);
f1s = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
linearPredictions = [];
predictionCat = [];

%outerloop
for i = 1:10
    fprintf("Fold %d\n",i);
    % retrieve training and test subdatasets
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  
    fprintf("InnerLoop Start of Fold %d\n",i);
    % Do inner cross validation
    currentModelParameters = innerLoop(trainingInputs,trainingTargets,5);

    % train the model
    fprintf("creating model of Fold %d\n",i);
    fprintf("p-order: %d, constraint: %d",currentModelParameters.qValue,currentModelParameters.cValue);
    %currentModel = fitcsvm(trainingInputs,trainingTargets, 'Standardize',true,'KernelFunction', 'linear', 'BoxConstraint', currentModelParameters.cValue); 
    currentModel = fitcsvm(trainingInputs,trainingTargets, 'Standardize',true,'KernelFunction', 'polynomial', 'PolynomialOrder',currentModelParameters.qValue, 'BoxConstraint',currentModelParameters.cValue);
    % SVM = fitcsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', sigmaValues(sigmaIndex), 'BoxConstraint', cValues(cIndex));

    %calculate f1 of this fold
    [predictions, score] = predict(currentModel,testingInputs);
    confusion = confusion_matrix(predictions, testingTargets);
    best_C_values (i) = currentModelParameters.cValue;
    recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
    precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
    accuracy(i) =(confusion(1,1)+confusion(2,2))/sum(sum(confusion));
    f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));


    % result
    predictionsMatrix{i} = predictions;
    linearPredictions = vertcat(linearPredictions, predictions(:,1));
    scoreMatrix{i} = score;
    supportVectors(i) = size(currentModel.SupportVectors,1);

end

% concatenate predictions
predictionCat = vertcat(predictionCat, predictionsMatrix{:});


%inner loop function
function [bestHyperParameter] = innerLoop(inputs,targets,k)

    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 4050
    foldLength = size(inputs,1)/k;          % foldLength = 450
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point

    iteration = 1;

    hyperparameterInformation = cell(1,5);

    %Box constraint values
    cValues = [0.01, 0.05, 0.1, 0.5 , 1 , 5 , 10, 50, 100, 500, 1000];

    %polynomial order
    pOrder = [2,3,4];

    for pIndex=1:length(pOrder)

        for cIndex=1:length(cValues)

            fprintf("InnerLoop c:value: %d, polyOrder : %d\n",cValues(cIndex),pOrder(pIndex));
            f1s = zeros(1,k);
            recalls = zeros(1,k);
            precisions = zeros(1,k);
            accuracy = zeros (1,k);

            for i=1:k     % each iteration performs one time of training and CV

                % retrieve training and testing dataset for i fold
                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  

                % training SVM

                SVM = fitcsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', pOrder(pIndex), 'BoxConstraint', cValues(cIndex));
                sv = size(SVM.SupportVectors,1);

                % prediction and evaluation. 

                predictions = predict(SVM,testingInputs);
                confusion = confusion_matrix(predictions, testingTargets);
                recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
                precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
                accuracy(i) =(confusion(1,1)+confusion(2,2))/sum(sum(confusion));
                f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));

            end     % end of fold cross validation 

            % determine mean f1 value
            f1 = mean(f1s);

            % store hyperparameters into a structure
            struct.cValue = cValues(cIndex);
            struct.qValue = pOrder(pIndex);
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);
            struct.f1 = f1;
            struct.f1s = f1s;
            struct.accuracy = accuracy;
            hyperparameterInformation{iteration} = struct;

            iteration = iteration + 1;

        end

    end



    % select the best f1
    bestf1 = 0;

    for i=1:length(hyperparameterInformation)

        currentf1 = hyperparameterInformation{i}.f1;

        if currentf1 > bestf1

            bestf1 = hyperparameterInformation{i}.f1;
            bestHyperParameter = hyperparameterInformation{i};

        end

    end

end


% Confusion matrix
function cm = confusion_matrix(outputs, labels)

% cm = zeros(2);
tp=0;tn=0;fp=0;fn=0;

for i=1:length(outputs)
    if (labels(i,1) == 1) && (outputs(i,1)==1)
        tp=tp+1;
    elseif (labels(i,1) == 0) && (outputs(i,1)==0)
        tn=tn+1;
    elseif (labels(i,1) == 1) && (outputs(i,1)==0)
        fn=fn+1;
    else
        fp=fp+1;
    end
end

cm = [tp, fn; fp, tn];

end

% CV Partition
function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets)

    validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data

    % the remaining are indexes of training data
    if i==1
        trainPerm = P(foldLength+1:end);
    elseif i==k
        trainPerm = P(1:numOfExamples);
    else
        trainPerm1 = P(1:(i-1)*foldLength);
        trainPerm2 = P(i*foldLength+1:end);
        trainPerm = [trainPerm1,trainPerm2];
    end

    % Set up Division of Data for Training, Validation, Testing
    % find the values of features and labels with their corresponding indexes
    trainingTargets = targets( trainPerm, :);
    trainingInputs = inputs( trainPerm, :);

    % find the values of features and labels with their corresponding indexes
    testingTargets = targets( validPerm, :);
    testingInputs = inputs( validPerm, :);

end

