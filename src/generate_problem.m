function [Q, q, l, u, a, b, x_start] = generate_problem(n, scale, intersection_percentage, actv_percentage)
    % docstring

    [l, u, a, b] = random_constraints(n, scale, intersection_percentage);

    while any(l > u) || a'*u < b
        [l, u, a, b] = random_constraints(n, scale, intersection_percentage);
    end

    is_invertible_Q = actv_percentage ~= -1;
    
    A = randn(n, n)*scale;
    while is_invertible_Q
        A = randn(n, n)*scale;
        if rank(A) == n; break; end    % will be true nearly all the time
    end

    Q = A'*A;

    if is_invertible_Q
        % generate q- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        %
        % We first generate the unconstrained minimum z of the problem in the form
        %
        %    min_x (1/2) ( x - z )^T * Q * ( x - z ) =
        %          (1/2) x^T * Q * x - z^T * Q * x + (1/2) z^T * Q * z
        %
        % and then we set q = - z^T Q

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

    x_start = projection(l, u, a, b, randn(n, 1)*scale, 1e-15);
end


function [l, u, a, b] = random_constraints(n, scale, percentage)
    % docstring

    a = rand(n, 1)*scale;    
    l = rand(n, 1)*(scale/2);
    u = l + rand(n, 1)*(scale/2);
    
    b_min = a'*l;
    b_max = a'*u;
        
    offset_b = (b_max - b_min)*((1 - percentage)); 
    
    b_min = b_min + offset_b;
    
    b = b_min + rand()*(b_max - b_min);
end
