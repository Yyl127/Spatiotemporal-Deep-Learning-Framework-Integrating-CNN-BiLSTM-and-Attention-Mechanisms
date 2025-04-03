classdef SpatioTemporalAttention1D < nnet.layer.Layer
    properties (Learnable)
        W_spatial   % [C,1,T] Spatial weights independent at each time step
        W_temporal  % [C,1,T] Time weights independent for each channel
        alpha       % Learnable scaling factor
    end
    
    methods
        function layer = SpatioTemporalAttention1D(name)
            layer.Name = name;
            layer.alpha = dlarray(0.1); 
        end
        
        function Z = predict(layer, X)
         

            spatial_att = sigmoid( sum(X .* layer.W_spatial, 2, 'native') ); % [C,1,B,T]
            
            temporal_att = sigmoid( sum(X .* layer.W_temporal, [1,3], 'native') ); % [C,1,B,T]
       
            combined_att = spatial_att .* temporal_att;
  
            Z = X + layer.alpha * (X .* combined_att);
        end
        
        function layer = initialize(layer, X)
            [C, ~, ~, T] = size(X);
            
        
            layer.W_spatial = dlarray(randn(C,1,T) * sqrt(2/(C+T))); 
            layer.W_temporal = dlarray(randn(C,1,T) * sqrt(2/(C+T)));
            layer.alpha = dlarray(0.1); 
        end
    end
end
