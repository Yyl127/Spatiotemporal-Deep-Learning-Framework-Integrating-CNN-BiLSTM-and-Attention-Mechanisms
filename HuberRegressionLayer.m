classdef HuberRegressionLayer < nnet.layer.RegressionLayer
    properties
        Delta 
    end
    
    methods
        function layer = HuberRegressionLayer(delta, name)
            layer.Delta = delta;
            layer.Name = name;
            layer.Description = 'Huber loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            error = Y - T;
            quadratic = abs(error) <= layer.Delta;
            loss = sum(0.5 * quadratic .* (error.^2) + ...
                      (1 - quadratic) .* (layer.Delta * abs(error) - 0.5 * layer.Delta^2));
            loss = loss / size(Y, 4); 
        end
    end
end