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
table.census_income = table.census_income - 1;

%% Data preparation for cross-validation
k=10; % k-fold
sampleSize = 500; % size of each set divisible by 10
lastn = 25162; 
table = table(1:end-lastn,:); %remove last 2 rows
x = table2array(table); %convert table to array
y = table.census_income;
[r, ~] = size(x);
numTestItems = round(r*0.1); %size of test set
numTrainingItems = r - numTestItems; % leftover to be training set
dataIndices = randperm(r); % shuffle the dataset 
shuffled_data = x(dataIndices,:);

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
    
    currentModel = fitcsvm(trainingData_x,trainingData_y, 'Standardize',true,'KernelFunction', 'linear', 'BoxConstraint', 1); 
    
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

    predictionsMatrix{fold} = predictions;
    scoreMatrix{fold} = score;
    supportVectors(fold) = size(currentModel.SupportVectors,1);
end

predictionCat = vertcat(predictionCat, predictionsMatrix{:});

% Confusion matrix
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