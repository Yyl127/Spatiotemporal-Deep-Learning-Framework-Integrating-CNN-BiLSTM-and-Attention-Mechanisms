function [output1,output2,output3,output4,output5,output6,output7,output8,output9,output10]= CNN_BiLSTM_BayesOpt(inputArg1)

Data=inputArg1;


Data(find(isinf(Data)==1)) = 0;
Data(find(isnan(Data)==1)) = 0;

row_to_delete = []; 
for i = 1:size(Data, 1)
    if sum(Data(i, 1:10) == 0) >= 8
        row_to_delete = [row_to_delete; i];
    end
end
Data(row_to_delete, :) = [];

%%Partitioning of Training and Testing Datasets
TotalSamples = size(Data, 1); 
InPut_num = 1:11;  
OutPut_num = 12;   
Temp = randperm(TotalSamples);

Train_Size = round(0.7 * TotalSamples);  
Train_InPut = Data(Temp(1:Train_Size), InPut_num);  
Train_OutPut = Data(Temp(1:Train_Size), OutPut_num);  
Test_InPut = Data(Temp(Train_Size+1:end), InPut_num);  
Test_OutPut = Data(Temp(Train_Size+1:end), OutPut_num);  
M = size(Train_InPut, 1);
N = size(Test_InPut, 1);

clear Temp TotalSamples Train_Size;
%% Data Normalization
[~, Ps.Input] = mapminmax([Train_InPut;Test_InPut]', -1, 1); 
Train_InPut = mapminmax('apply',Train_InPut',Ps.Input);
Test_InPut = mapminmax('apply',Test_InPut',Ps.Input);
[~, Ps.Output] = mapminmax([Train_OutPut;Test_OutPut]', -1, 1);
Train_OutPut = mapminmax('apply',Train_OutPut',Ps.Output);
Test_OutPut = mapminmax('apply',Test_OutPut',Ps.Output);

Temp_TrI = cell(size(Train_InPut,2),1);
Temp_TrO = cell(size(Train_OutPut,2),1);
Temp_TeI = cell(size(Test_InPut,2),1);
Temp_TeO = cell(size(Test_OutPut,2),1);

for i = 1:size(Train_InPut,2)
    Temp_TrI{i} = Train_InPut(:,i);
    Temp_TrO{i} = Train_OutPut(:,i);
end
Train_InPut = Temp_TrI;
Train_OutPut = Temp_TrO;

for i = 1:size(Test_InPut,2)
    Temp_TeI{i} = Test_InPut(:,i);
    Temp_TeO{i} = Test_OutPut(:,i);
end
Test_InPut = Temp_TeI;
Test_OutPut = Temp_TeO;

clear Temp_TrI Temp_TrO Temp_TeI Temp_TeO;

%% Bayesian Optimization Parameter Configuration
numFeatures = length(InPut_num);
numResponses = length(OutPut_num);

% Defining the Hyperparameter Search Space
optimVars = [
    optimizableVariable('numHiddenUnits', [100, 1000], 'Type', 'integer')
    optimizableVariable('LearnRate', [1e-5, 1e-2], 'Transform', 'log')
    optimizableVariable('filterSize', [16, 64], 'Type', 'integer')
    optimizableVariable('numFilters', [64, 256], 'Type', 'integer')
    optimizableVariable('dorp_rate', [0.1, 0.5])
];

% Creating the Objective Function
bayesfun = @(params) bayesoptObjective(params, Train_InPut, Train_OutPut, Test_InPut, Test_OutPut,...
                                      numFeatures, numResponses, Ps);

% Running Bayesian Optimization
results = bayesopt(bayesfun, optimVars, ...
    'MaxObjectiveEvaluations', 100, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
     'PlotFcn', [],...     
    'UseParallel', false);

% Obtain the Optimal Parameters
bestParams = results.XAtMinObjective;

%% Train the Final Model Using the Optimal Parameters
numHiddenUnits = bestParams.numHiddenUnits;
LearnRate = bestParams.LearnRate;
filterSize = bestParams.filterSize;
numFilters = bestParams.numFilters;
dorp_rate = bestParams.dorp_rate;
Train_number = 200;  

%% Construct the Optimal Network
inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input');
gaussianNoiseLayer = GaussianNoiseLayer(0.1, 'noise_layer');

% CNN
cnnBranch = [
    convolution1dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    layerNormalizationLayer('Name', 'norm1')
    dropoutLayer(dorp_rate)
    convolution1dLayer(floor(filterSize/2), floor(numFilters/2), 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    layerNormalizationLayer('Name', 'norm2')
    maxPooling1dLayer(2, 'Padding', 'same', 'Name', 'pool')
    flattenLayer('Name', 'flatten')
    SCAttentionLayer('sc_attention')
    fullyConnectedLayer(50, 'Name', 'fc_cnn')
];

% BiLSTM
lstmBranch = [
    bilstmLayer(numHiddenUnits, 'Name', 'bilstm')
    dropoutLayer(dorp_rate)
    fullyConnectedLayer(50, 'Name', 'fc_lstm')
];

% Merge the Branches
mergeBranch = [
    concatenationLayer(1, 2, 'Name', 'concat')
    SpatioTemporalAttention('st_att')
    fullyConnectedLayer(numResponses, 'Name', 'fc_final')
    HuberRegressionLayer(1.0, 'huber_loss')
];

lgraph = layerGraph(inputLayer);
lgraph = addLayers(lgraph, gaussianNoiseLayer);
lgraph = addLayers(lgraph, cnnBranch);
lgraph = addLayers(lgraph, lstmBranch);
lgraph = addLayers(lgraph, mergeBranch);

lgraph = connectLayers(lgraph, 'input', 'noise_layer');
lgraph = connectLayers(lgraph, 'noise_layer', 'conv1');
lgraph = connectLayers(lgraph, 'noise_layer', 'bilstm');
lgraph = connectLayers(lgraph, 'fc_cnn', 'concat/in1');
lgraph = connectLayers(lgraph, 'fc_lstm', 'concat/in2');

%% Train the Final Model
options = trainingOptions('adam', ...
    'MaxEpochs', Train_number, ...
    'GradientThreshold', 1, ...
    'ValidationData', {Test_InPut, Test_OutPut}, ...
    'ValidationFrequency', 30, ...
    'InitialLearnRate', LearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', floor(Train_number/2), ...
    'LearnRateDropFactor', 0.5, ...
    'L2Regularization', 0.001, ...
    'Verbose', 1, ...
    'Plots','none');

net = trainNetwork(Train_InPut, Train_OutPut, lgraph, options);

%% Network Testing and Evaluation
TPred = predict(net,Train_InPut);
YPred = predict(net,Test_InPut);


%% Inverse Normalization
True_Train = []; 
Predict_Train = []; 
True_Test = []; 
Predicte_Test = []; 

for i = 1:size(Train_InPut,1)
    True_Train = [True_Train,mapminmax('reverse',Train_OutPut{i},Ps.Output)];
    Predict_Train = [Predict_Train,mapminmax('reverse',TPred{i},Ps.Output)];
end
Predict_Train = double(Predict_Train);


for i = 1:size(Test_OutPut,1)
    True_Test = [True_Test,mapminmax('reverse',Test_OutPut{i},Ps.Output)];
    Predicte_Test = [Predicte_Test,mapminmax('reverse',YPred{i},Ps.Output)];
end


RMSE = sqrt(mean((True_Test - Predicte_Test).^2));
R = 1 - norm(True_Test - Predicte_Test)^2 / norm(Predicte_Test - mean(True_Test))^2;
mae = mean(abs(True_Test - Predicte_Test));
mbe = mean(True_Test - Predicte_Test);

R1 = 1 - norm(True_Train - Predict_Train)^2 / norm(Predict_Train - mean(True_Train))^2;
R2 = 1 - norm(True_Test  - Predicte_Test)^2 / norm(Predicte_Test  - mean(True_Test ))^2;

RMSE1 = sqrt(mean((True_Train-Predict_Train).^2));
RMSE2 = sqrt(mean((True_Test-Predicte_Test).^2));


% MAE
mae1 = sum(abs(Predict_Train - True_Train), 2)' ./ length(True_Train);
mae2 = sum(abs(Predicte_Test - True_Test ), 2)' ./ length(True_Test);

mbe1 = sum((Predict_Train - True_Train), 2)' ./ length(True_Train);
mbe2 = sum((Predicte_Test - True_Test ), 2)' ./ length(True_Test);


%% Bayesian Optimization Objective Function
function valRMSE = bayesoptObjective(params, Train_InPut, Train_OutPut, Test_InPut, Test_OutPut,...
                                    numFeatures, numResponses, Ps)
try

    inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input');
    gaussianNoiseLayer = GaussianNoiseLayer(0.1, 'noise_layer');
    
    % CNN
    cnnBranch = [
        convolution1dLayer(params.filterSize, params.numFilters, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        layerNormalizationLayer('Name', 'norm1')
        dropoutLayer(params.dorp_rate)
        convolution1dLayer(floor(params.filterSize/2), floor(params.numFilters/2), 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        layerNormalizationLayer('Name', 'norm2')
        maxPooling1dLayer(2, 'Padding', 'same', 'Name', 'pool')
        flattenLayer('Name', 'flatten')
        SCAttentionLayer('sc_attention')
        fullyConnectedLayer(50, 'Name', 'fc_cnn')
    ];
    
    % BiLSTM
    lstmBranch = [
        bilstmLayer(params.numHiddenUnits, 'Name', 'bilstm')
        dropoutLayer(params.dorp_rate)
        fullyConnectedLayer(50, 'Name', 'fc_lstm')
    ];
    
    
    mergeBranch = [
        concatenationLayer(1, 2, 'Name', 'concat')
        SpatioTemporalAttention('st_att')
        fullyConnectedLayer(numResponses, 'Name', 'fc_final')
        HuberRegressionLayer(1.0, 'huber_loss')
    ];
    
   
    lgraph = layerGraph(inputLayer);
    lgraph = addLayers(lgraph, gaussianNoiseLayer);
    lgraph = addLayers(lgraph, cnnBranch);
    lgraph = addLayers(lgraph, lstmBranch);
    lgraph = addLayers(lgraph, mergeBranch);
    
    lgraph = connectLayers(lgraph, 'input', 'noise_layer');
    lgraph = connectLayers(lgraph, 'noise_layer', 'conv1');
    lgraph = connectLayers(lgraph, 'noise_layer', 'bilstm');
    lgraph = connectLayers(lgraph, 'fc_cnn', 'concat/in1');
    lgraph = connectLayers(lgraph, 'fc_lstm', 'concat/in2');
    
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 15,...  
        'GradientThreshold', 1, ...
        'ValidationData', {Test_InPut, Test_OutPut}, ...
        'ValidationFrequency', 30, ...
        'InitialLearnRate', params.LearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 25, ...
        'LearnRateDropFactor', 0.5, ...
        'L2Regularization', 0.001, ...
        'Verbose', false, ...
        'Plots','none');
    
    
    net = trainNetwork(Train_InPut, Train_OutPut, lgraph, options);
    YPred = predict(net, Test_InPut);
    Predicte_Test = [];
    True_Test = [];
    for i = 1:size(Test_OutPut,1)
        True_Test = [True_Test, mapminmax('reverse', Test_OutPut{i}, Ps.Output)];
        Predicte_Test = [Predicte_Test, mapminmax('reverse', YPred{i}, Ps.Output)];
    end
    Predicte_Test = double(Predicte_Test);
    True_Test = double(True_Test);
  
    valRMSE = sqrt(mean((True_Test - Predicte_Test).^2));
catch
    valRMSE = 1e4;  
end
end

output1=R1;
output2=R2;
output3=Ps;
output4=net;

output5=RMSE1;
output6=RMSE2;
output7=mae1;
output8=mae2;

output9=mbe1;
output10=mbe2;
end
