%% data loading and feature selection
table = readtable('adult.csv');

%attributes removed due to low predictor importance
%table = removevars(table, [1 2 3 4 9 10 12 13 14]);

attribute_name = table.Properties.VariableNames;

%% Label encoding for table
table.workclass = double(categorical(table.workclass));
table.education = double(categorical(table.education));
table.marital_status = double(categorical(table.marital_status));
table.occupation = double(categorical(table.occupation));
table.relationship = double(categorical(table.relationship));
table.race = double(categorical(table.race));
table.sex = double(categorical(table.sex));
table.sex = table.sex -1;
table.native_country = double(categorical(table.native_country));
table.census_income = double(categorical(table.census_income));
table.census_income = table.census_income - 1; %convert values 1,2 to 0,1 for easier processing

%% Data preparation for cross-validation
k=10; % k-fold
sampleSize = 300; % size of each set divisible by 10
lastn = 27162; 
table = table(1:end-lastn,:); %remove last 2 rows
x = table2array(table); %convert table to array
y = table.census_income;
[r, ~] = size(x);
numTestItems = round(r*0.1); %size of test set
numTrainingItems = r - numTestItems; % leftover to be training set
dataIndices = randperm(r); % shuffle the dataset 
shuffled_data = x(dataIndices,:);

%% Global Variable
global doDisp;  
doDisp = false;  % display flag

iteration = 1;  % initial iteration
%% Empty variables initialization
recalls = zeros(1,10);
precisions = zeros(1,10);
f1s = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
predictionCat = [];


%% K-Fold cross validation
for fold =1:k
    fprintf(" %d Fold\n",fold);
    test_indices = 1+(fold-1)*sampleSize:fold*sampleSize;
    train_indices = [1:(fold-1)*sampleSize, fold*sampleSize+1:numTrainingItems];
     
    %% Training data preparation
    trainingData = shuffled_data(train_indices,:);
    testData = shuffled_data(test_indices,:);
    trainingData_x = trainingData(:,1:14);
    trainingData_y = trainingData(:,15);
    
    %% Inner-fold Cross Validation
    currentModelParameters = innerLoop(trainingData,3);
    
    %% SVM training
    currentModel = fitcsvm(trainingData_x,trainingData_y, 'Standardize',true,'KernelFunction', 'polynomial', 'PolynomialOrder',currentModelParameters.pOrder, 'BoxConstraint',currentModelParameters.cValue); 
    
    %% Test data preparation
    test_x = testData(:,1:14); % test data attributes
    test_y = testData(:,15); %test data targets
    
    [predictions, score] = predict(currentModel,test_x);
    confusion = confusion_matrix(predictions, test_y);
    recalls(fold) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
    precisions(fold) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
    f1s(fold) = 2*((precisions(fold)*recalls(fold))/(precisions(fold)+recalls(fold)));
    accuracy = (confusion(1,1)+confusion(2,2))/(confusion(1,1)+confusion(1,2)+confusion(2,1)+confusion(2,2));
    
    %% Accuracy storing and printing
    fprintf("accuracy = %.3f%%\n",accuracy);
    accuracy_list(1,fold) = accuracy; % store accuracy of each fold
    
    
    %% Result
    predictionsMatrix{fold} = predictions;
    scoreMatrix{fold} = score;
    supportVectors(fold) = size(currentModel.SupportVectors,1);
end

predictionCat = vertcat(predictionCat, predictionsMatrix{:});


%% Inner-fold cross validation
function [bestHyperParameter] = innerLoop(inputs,j)

    [r, ~] = size(inputs);
    innerSample = round(r/j);
    numTestItems2 = round(r*0.1); %size of test set
    numTrainingItems2 = r - numTestItems2; % leftover to be training set
    dataIndices2 = randperm(r); % shuffle the dataset 
    shuffled_data2 = inputs(dataIndices2,:);
    
    iteration = 1;

    hyperparameterInformation = cell(1,5);

    %Box constraint values
    cValues = [0.1, 0.5 , 1 , 5 , 10, 50];

    %polynomial order
    pOrder = [2,3,4];

    for pIndex=1:length(pOrder)

        for cIndex=1:length(cValues)

            f1s = zeros(1,j);
            recalls = zeros(1,j);
            precisions = zeros(1,j);

            for i=1:j   
                
                test_indices2 = 1+(j-1)*innerSample:j*innerSample;
                train_indices2 = [1:(j-1)*innerSample, j*innerSample+1:numTrainingItems2];
                
                trainingData2 = shuffled_data2(train_indices2,:);
                testData2 = shuffled_data2(test_indices2,:);
                trainingData_x = trainingData2(:,1:14);
                trainingData_y = trainingData2(:,15);
                
                % training SVM
                SVM = fitcsvm(trainingData_x, trainingData_y, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', pOrder(pIndex), 'BoxConstraint', cValues(cIndex));
                sv = size(SVM.SupportVectors,1);
                
                test_x = testData2(:,1:14); % test data attributes
                test_y = testData2(:,15); %test data targets
              
                % prediction and evaluation. 
                predictions = predict(SVM,test_x);
                confusion = confusion_matrix(predictions, test_y);
                recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
                precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
                f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));

            end     % end of fold cross validation 

            % determine mean f1 value
            f1 = mean(f1s);

            % store hyperparameters into a structure
            struct.pOrder = pOrder(pIndex);
            struct.cValue = cValues(cIndex);
            %struct.sigmaValue = sigmaValues(sigmaIndex);
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);
            struct.f1 = f1;
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
%% Confusion matrix
function cm = confusion_matrix(outputs, labels)

% cm = zeros(2);
tp=0;
tn=0;
fp=0;
fn=0;

for i=1:length(outputs)
    if (labels(i) == 1) && (outputs(i)==1)
        tp=tp+1;
    elseif (labels(i) == 0) && (outputs(i)==0)
        tn=tn+1;
    elseif (labels(i) == 1) && (outputs(i)==0)
        fn=fn+1;
    else
        fp=fp+1;
    end
end

cm = [tp, fn; fp, tn];

end

