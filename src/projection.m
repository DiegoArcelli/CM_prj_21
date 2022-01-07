function [x_proj] = projection(a, l, u, b, y, eps)
    % docstring
    l_tilde = l - y;
    u_tilde = u - y;
    a_tilde = -a;
    b_tilde = -(b - a'*y);
    
    % handle when a_tilde = 0
    mi_plus = -2*(l_tilde./a_tilde);
    mi_minus = -2*(u_tilde./a_tilde);

    % use quick sort O(n^2) | avg O(n log n)
    break_points = union(mi_plus, mi_minus);
    
    q_p = @(m) lagrangian_dual_prime(a_tilde, l_tilde, u_tilde, b_tilde, m);
    
    mu_l = 0;
    mu_u = 0;
    g_u = q_p(break_points(end));
    g_l = q_p(break_points(1));
    
    disp(q_p(break_points(1)))
    
    while ~isempty(break_points)
        j = ceil(length(break_points)/2);
        mu = break_points(j);
        
        if abs(q_p(mu)) < eps
            x_proj = min_x_mu(l_tilde, u_tilde, mu) + y;
            return;
        elseif q_p(mu) < 0
            mu_u = mu;
            g_u = q_p(mu);
            break_points = break_points(break_points<mu);
        else
            mu_l = mu;
            g_l = q_p(mu);
            break_points = break_points(break_points>mu);
        end
    end
    
    mu_star = mu_l + g_u*(mu_u - mu_l)/(g_u - g_l);
    x_proj = min_x_mu(l_tilde, u_tilde, mu_star) + y;
end


function [x_m] = min_x_mu(l, u, mu)
    % docstring
    x_m = median([l, u, repmat(-mu, size(u))], 2);
end


function [q_prime] = lagrangian_dual_prime(a, l, u, b, mu)
    % docstring
    x_m = min_x_mu(l, u, mu);
    q_prime = -b + a'*x_m;
end