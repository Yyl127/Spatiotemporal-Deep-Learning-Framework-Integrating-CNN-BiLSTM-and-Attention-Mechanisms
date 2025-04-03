clear;
load TWS_data.mat
data=merged_matrix;
load predict_data.mat
data_2=pre_data;
a=[7,11,12,12,12,12,12,12,12,10,10,9,9,9,9,5,5,12,12,12,12,12,0];
z=a.*156;
C = zeros(22,9);
pre_data=zeros(120,130,261);
x=1;
y=1092;
i=1;
while i<23
[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10]= CNN_BiLSTM_BayesOpt(data(x:y,:));
C(i,1)=output1;
C(i,2)=output2;
C(i,3)=output5;
C(i,4)=output6;
C(i,5)=output7;
C(i,6)=output8;
C(i,7)=output9;
C(i,8)=output10;

C(i,9)=y-x+1;
x=x+z(i);
y=y+z(i+1);
Ps=output3;
net=output4;
                if i==1
                for j=1:9
                    new_data=data_2(:,:,(i-1)*12+j);
                   New_Input = mapminmax('apply', new_data', Ps.Input);                      
                    Temp_NewI = cell(size(New_Input, 2), 1);
                    for k = 1:size(New_Input, 2)
                        Temp_NewI{k} = New_Input(:, k);
                    end
                    New_Input = Temp_NewI;
                    YPred_New = predict(net, New_Input);       
                    Predicted_New = [];
                    for n = 1:size(YPred_New, 1)
                        Predicted_New = [Predicted_New, mapminmax('reverse', YPred_New{n}, Ps.Output)];
                    end
                    Predicted_New = double(Predicted_New);  
                    [output]= to3(Predicted_New);
                    pre_data(:,:,j)=output;
                   
                    
                end
            else
                for j=1:12
                    new_data=data_2(:,:,(i-1)*12+j-3);
        
                    New_Input = mapminmax('apply', new_data', Ps.Input);                      

                    Temp_NewI = cell(size(New_Input, 2), 1);
                    for k = 1:size(New_Input, 2)
                        Temp_NewI{k} = New_Input(:, k);
                    end
                    New_Input = Temp_NewI;

                    
                    YPred_New = predict(net, New_Input);  

                   
                    Predicted_New = [];
                    for n = 1:size(YPred_New, 1)
                        Predicted_New = [Predicted_New, mapminmax('reverse', YPred_New{n}, Ps.Output)];
                    end
                    Predicted_New = double(Predicted_New); 
                    [output]= to3(Predicted_New);
                    pre_data(:,:,(i-1)*12+j-3)=output;
                   
                    
                end
            end
                  end

end


