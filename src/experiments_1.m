% FIXED STEP SIZE

% 2 dimensional problems
n = 2;


% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5, 0);
    
    f = @(x) objective_function(Q,q,x);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(f, l, u, a, b , x_start, 10e-16, 10e-16, 1000, "fixed", 1/L, 0, 0);
    toc();

    tic();
    [x_star_real, f_star_real] = minimize_matlab_kqp(n, x_start, Q, q, l, u, a, b);
    toc();

    disp(norm(x_star - x_star_real)/norm(x_star_real));
    disp(norm(f_star - f_star_real)/norm(f_star_real));
end

% plot the convergence curve
plot(vecnorm(g_s)); 

input("");

% 10 dimensional problems
n = 10;

% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5, 0);
    
    f = @(x) objective_function(Q,q,x);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(f, l, u, a, b , x_start, 10e-16, 10e-16, 1000, "fixed", 1/L, 0, 0);
    toc();

    tic();
    [x_star_real, f_star_real] = minimize_matlab_kqp(n, x_start, Q, q, l, u, a, b);
    toc();

    disp(norm(x_star - x_star_real)/norm(x_star_real));
    disp(norm(f_star - f_star_real)/norm(f_star_real));
end

% plot the convergence curve
plot(vecnorm(g_s));

input("");

% 100 dimensional problems
n = 124;

% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5, 0);
    
    f = @(x) objective_function(Q,q,x);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(f, l, u, a, b , x_start, 10e-15, 10e-15, 1000, "fixed", 1/L, 0, 0);
    toc();

    tic();
    [x_star_real, f_star_real] = minimize_matlab_kqp(n, x_start, Q, q, l, u, a, b);
    toc();

    disp(norm(x_star - x_star_real)/norm(x_star_real));
    disp(norm(f_star - f_star_real)/norm(f_star_real));
end

% plot the convergence curve
plot(vecnorm(g_s));
