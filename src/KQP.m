function [x_star, f_star] = KQP(Q, q, x_start,a, l, u, b, eps, eps_prime, stepspize)
    
    x_i = x_start;
    x_prev = Inf(size(x_start));

    f = @(x) objective_function(Q,q,x);
    f_i = 0;
    while norm(x_i - x_prev) >= eps
        [f_i, g_i] = f(x_i);
        d = -g_i;
        alpha = stepspize;
        y = x_i + alpha*d;
        x_prev = x_i;
        disp(y);
        x_i = projection(a, l, u, b, y, eps_prime);
        disp(x_i);
        disp(" ");
    end
    x_star = x_i;
    f_star = f_i;
end