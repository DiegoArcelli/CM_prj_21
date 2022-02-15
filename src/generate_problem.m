function [Q, q, l, u, a, b, x_start] = generate_problem(n, scale, intersection_percentage, actv_percentage)
    % generate a random problem which can be solved by KQP.m
    %
    % input arguments:
    % - n: the dimension of the problem 
    % - scale: scalar to scale the values of the values of the generated
    % parameters
    % - intersection_percentage: percentage of intersection between the
    % regions defined by a'x >= b and l <= x <= u
    % - actv_percentage: tune how much the optimum point is outside from the
    % beasible region (if Q is invertible)
    %
    % outputs:
    % - Q (a n x n positive semi-definite matrix) and q (a n dimensional vector) to
    % represent the quadratic function to minimize
    % - l, u, a (n dimensional vectors) and b (scalar) to define the feasible region
    % - x_start: the starting point of the problem

    [l, u, a, b] = random_constraints(n, scale, intersection_percentage);

    is_invertible_Q = actv_percentage ~= -1;
    
    A = randn(n, n)*scale;
    while is_invertible_Q
        A = randn(n, n)*scale;
        if rank(A) == n; break; end    % will be true nearly all the time
    end

    Q = A'*A;

    if is_invertible_Q
        % taken from: https://elearning.di.unipi.it/pluginfile.php/47170/mod_resource/content/2/genBCQP.m
        % generate q- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        z = zeros( n , 1 );

        % outb( i ) = true if z( i ) will be out of the bounds
        outb = rand( n , 1 ) <= (1 - actv_percentage);

        % 50/50 chance of being left of lb or right of ub
        lr = rand( n , 1 ) <= 0.5;
        left = outb & lr;
        right = outb & ~ lr;

        % a random amount left of the lb (0)
        z( left ) = l( left ) .* (1 - rand( sum( left ) , 1 ));

        % a random amount right of the ub (u)
        z( right ) = u( right ) .* (1 + rand( sum( right ) , 1 ));

        outb = ~outb;  % entries that will be inside the bound
        % pick at random in [ l , u ]
        z( outb ) = l(outb) +  rand( sum( outb ) , 1 ) .* (u( outb ) - l( outb ));

        q = - Q * z;
    else
        q = randn(n, 1)*scale;
    end

    x_start = projection(l, u, a, b, randn(n, 1)*scale, 10e-10, false);
end


function [l, u, a, b] = random_constraints(n, scale, percentage)
    % generate random parameters for the constraints
    % - n: dimension of the problem
    % - scale: scalar to scale the values of the values of the generated
    % parameters
    % - percentage:  percentage of intersection between the
    % regions defined by a'x >= b and l <= x <= u
    % 
    % outputs:
    % - l, u, a (n dimensional vectors) and b (scalar) to define the feasible region

    a = rand(n, 1)*scale;
    
    l = rand(n, 1)*(scale/2);
    
    u = repmat(-1, size(l));
    while any(l > u)
        u = l + rand(n, 1)*(scale/2);
    end
    
    b_min = a'*l;
    b_max = a'*u;
        
    offset_b = (b_max - b_min)*(1 - percentage);
    
    b_min = b_min + offset_b;
    
    b = b_min + rand()*(b_max - b_min);
end
