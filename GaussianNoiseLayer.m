classdef GaussianNoiseLayer < nnet.layer.Layer
    properties (Learnable)
        StdDev % Noise Standard Deviation
    end
    
    methods
        function layer = GaussianNoiseLayer(stddev, name)
                        layer.StdDev = stddev;
            layer.Name = name;
        end
        
        function Z = predict(layer, X)
            Z = X + layer.StdDev .* randn(size(X), 'like', X);
        end
    end
end