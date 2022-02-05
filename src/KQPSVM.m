classdef KQPSVM
    % KQPSVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        C
        step_size
        params_step_size
        
        alpha_gez
        d_svs
        X_svs
        b_o
    end
    
    methods
        function obj = KQPSVM(C, step_size, params_step_size)
            obj.C = C;
            obj.step_size = step_size;
            obj.params_step_size = params_step_size;
        end
        
        function obj = fit(obj, X, d)
            % METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            f = @(alpha) f_svm(alpha, d.*X);

            N = size(X);
            N = N(1);

            l = zeros([N, 1]);
            u = repmat(obj.C, [N, 1]);

            x_start = zeros([N, 1]);

            [alpha_star, ~, ~, ~, ~] = KQP(f, l, u, d, 0, x_start, 1e-15, 1e-15, 1000, obj.step_size, obj.params_step_size, 0, 1);

            svs_idx = alpha_star ~= 0;
            obj.alpha_gez = alpha_star(svs_idx);
            obj.d_svs = d(svs_idx);
            obj.X_svs = X(svs_idx, :);

            on_margin_sv_idx = 0 < alpha_star & alpha_star < obj.C;
            on_margin_sv = X(on_margin_sv_idx, :);
            on_margin_d = d(on_margin_sv_idx);

            obj.b_o = 0;

            % add the mean over the on margin vectors to have numerical stability
            N_prime = size(on_margin_sv);
            N_prime = N_prime(1);
            for i= 1:N_prime
                sv = on_margin_sv(i,:);
                d_sv = on_margin_d(i);
                obj.b_o = obj.b_o + (d_sv - (obj.alpha_gez'*(obj.d_svs.*obj.X_svs))*sv');
            end

            obj.b_o = obj.b_o/N_prime;
        end
        
        function preds = predict(obj, X)
 
            preds = sign((obj.alpha_gez'*(obj.d_svs.*obj.X_svs))*X' + obj.b_o);
            preds = preds';
        end
    end
end

