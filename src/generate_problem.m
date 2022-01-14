function [Q, q, l, u, a, b, x_start] = generate_problem(n, scale)
    % docstring

    A = randn(n, n)*scale;
    Q = A'*A;
    q = randn(n, 1)*scale;

    [l, u, a, b] = random_constraints(n, scale);

    while any(l > u) || a'*u < b
        [l, u, a, b] = random_constraints(n, scale);
    end

    x_start = projection(l, u, a, b, randn(n, 1)*scale, 10e-10);
end


function [l, u, a, b] = random_constraints(n, scale)
    % docstring

    l = rand(n, 1)*(scale/2);
    u = l + rand(n, 1)*(scale/2);
    a = rand(n, 1)*scale;
    b = rand(1, 1)*scale;
end