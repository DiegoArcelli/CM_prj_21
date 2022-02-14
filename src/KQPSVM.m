classdef KQPSVM
    % Class that implement the soft-margin SVM linear classifier
    properties
        C % C parameter of the soft-margin
        step_size % step function to be called from the quadratic programming solver
        params_step_size % parameters passed to the step_size function
        
        % properties necessary to do prediction
        alpha_svs % alpha element, lagrangian multiplier, associated to the support vectors
        d_svs % target associated to the support vectors
        X_svs % support vectors
        b_o % interceptor of the classificator
    end
    
    methods
        function obj = KQPSVM(C, step_size, params_step_size)
            obj.C = C;
            obj.step_size = step_size;
            obj.params_step_size = params_step_size;
        end
        
        function obj = fit(obj, X, d)
            % fit method, create the classifier, based on the training
            % data.
            % Use the quadratic problem solved based on the gradient
            % projection to calculate the lagrangian multiplier (alphas)
            % necessary to compute the separator hyperplane
            %
            % input arguments:
            % - X (matrix) the matrix of the training samples
            % - d (vector) the vector of the training samples' target
            % output arguments:
            % - the model trained
            
            f = @(alpha) f_svm(alpha, d.*X);

            N = size(X);
            N = N(1);

            l = zeros([N, 1]);
            u = repmat(obj.C, [N, 1]);

            x_start = zeros([N, 1]);

            [alpha_star, ~, ~, ~, ~] = KQP(f, l, u, d, 0, x_start, 1e-15, 1e-15, 1000, obj.step_size, obj.params_step_size, 0, 1);

            svs_idx = alpha_star ~= 0;
            obj.alpha_svs = alpha_star(svs_idx);
            obj.d_svs = d(svs_idx);
            obj.X_svs = X(svs_idx, :);

            on_margin_sv_idx = 0 < obj.alpha_svs & obj.alpha_svs < obj.C;
            on_margin_sv = X(on_margin_sv_idx, :);
            on_margin_d = d(on_margin_sv_idx);

            obj.b_o = 0;

            % add the mean over the on margin vectors to have numerical stability
            N_prime = size(on_margin_sv);
            N_prime = N_prime(1);
            for i= 1:N_prime
                sv = on_margin_sv(i,:);
                d_sv = on_margin_d(i);
                obj.b_o = obj.b_o + (d_sv - (obj.alpha_svs'*(obj.d_svs.*obj.X_svs))*sv');
            end

            obj.b_o = obj.b_o/N_prime;
        end
        
        function preds = predict(obj, X)
            % predict the X's rows target using the separator hyperplane
            % trained by the fit call of this class.
            %
            % input arguments:
            % - X (matrix) the matrix of the samples to predict the target
            % output arguments:
            % - preds (vector) the predicted targets of X's rows

            preds = sign((obj.alpha_gez'*(obj.d_svs.*obj.X_svs))*X' + obj.b_o);
            preds = preds';
        end
    end
end

