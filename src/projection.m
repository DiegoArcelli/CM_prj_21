function [x_proj] = projection(l, u, a, b, y, eps)
    % function which projects the vector y into the region given by 
    % a'x >= b and l <= x <= u, by solving the constraint minimization
    % problem (x-y)'I(x-y) given the above constraints, as knapsack
    % quadratic problem using the lagrangian function
    %
    % input parameters:
    % - l, u, a (n dimensional vectors) and b (scalar) to define the
    % feasible region
    % - y: the n dimensional vector to project into the feasible region
    % - eps: the precision required for the algorithm
    % output parameters:
    % -  x_proj: the projection of y into the feasible region 

    
    % redefine the parameters of the costraints to solve the 
    % write ~x = x - y and solve the problem of minimizing ~x'I~x~
    % given ~a'x - ~b <= 0, ~l -  <= 0 and ~x - ~u <= 0 
    l_tilde = l - y;
    u_tilde = u - y;
    a_tilde = -a;
    b_tilde = -(b - a'*y);

    % compute the breakpoints
    mi_plus = -2*(l_tilde./a_tilde);
    mi_minus = -2*(u_tilde./a_tilde);

    % sort the breakpoints using the quick sort O(n^2) | avg O(n log n)
    % and select only the breakpoints >= 0, since for the KKN conditions
    % has to be mu >= 0
    break_points = union(mi_plus, mi_minus);
    break_points = break_points(break_points>0);

    q_p = @(m) lagrangian_dual_prime(l_tilde, u_tilde, a_tilde, b_tilde, m);
    
    % check if a'x(mu) - b <= 0 for mu = 0
    if q_p(0) <= 0
        x_proj = min_x_mu(l_tilde, u_tilde, a_tilde, 0) + y;
        return;
    end

    % add 0 as breakpoint before all other to avoid degenerative cases
    break_points = [0, break_points'];

    mu_l = 0;
    mu_u = 0;
    g_u = 0;
    g_l = 0;

    % apply the bisection method to find the breakpoint mu* such that
    % a'x(mu*) - b = 0
    while ~isempty(break_points)
        % find the median breakpoint mu*
        j = ceil(length(break_points)/2);
        mu = break_points(j);

        % if a'x(mu*) - b = 0 return the projection
        if abs(q_p(mu)) < eps
            x_proj = min_x_mu(l_tilde, u_tilde, a_tilde, mu) + y;
            return;
        % if a'x(mu*) - b < 0 search the breakpoints on the left
        elseif q_p(mu) < 0
            mu_u = mu;
            g_l = q_p(mu);
            break_points = break_points(break_points<mu);
        % if a'x(mu*) - b < 0 search the breakpoints on the right
        else
            mu_l = mu;
            g_u = q_p(mu);
            break_points = break_points(break_points>mu);
        end
    end
    
    % if we didn't find mu* such that a'x(mu*) - b = 0, then we
    % have two breakpoints mu- and mu+ such that 
    % a'x(mu-) - b < 0 and a'x(mu+) - b > 0 for which we can compute
    % the exact linear interpolation
    mu_star = mu_l + g_u*(mu_u - mu_l)/(g_u - g_l);
    x_proj = min_x_mu(l_tilde, u_tilde, a_tilde, mu_star) + y;
end


function [x_m] = min_x_mu(l, u, a, mu)
    % input arguments:
    % - l, u, a: three n dimensional vectors
    % - m: a scalar value
    % 
    % output: 
    % - x_m a n dimensional vector where the j-th component is the median
    % element among  l_j, u_j, -a_j*mu/2

    x_m = median([l, u, -(a*mu)/2], 2);
end


function [q_prime] = lagrangian_dual_prime(l, u, a, b, mu)
    % the derivative with resepect to mu of the lagrangian function
    % q'(mu) = -b*mu + a'x(mu) 
    %
    % input arguments:
    % - l, u, a (n dimensional vectors) and b (scalar) to define the
    % feasible region
    % - mu: the value in which evaluate q'
    % 
    % output arguments:
    % - q_prime: the value of q' in mu

    x_m = min_x_mu(l, u, a, mu);
    q_prime = -b + a'*x_m;
end