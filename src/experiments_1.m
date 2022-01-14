% FIXED STEP SIZE

% 2 dimensional problems
n = 2;

options = optimoptions('fmincon','Display','off');

% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(Q, q, l, u, a, b , x_start, 10e-16, 10e-16, 1000, "fixed", 1/L, 0);
    toc();

    tic();
    x_mega = fmincon(@(x) objective_function(Q, q, x), x_start, [-eye(n); eye(n); -a'], [-l; u; -b], [], [], [], [], [], options);
    toc();

    disp(norm(x_star - x_mega)/norm(x_mega));
end

% 10 dimensional problems
n = 10;

% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(Q, q, l, u, a, b , x_start, 10e-16, 10e-16, 1000, "fixed", 1/L, 0);
    toc();

    tic();
    x_mega = fmincon(@(x) objective_function(Q, q, x), x_start, [-eye(n); eye(n); -a'], [-l; u; -b], [], [], [], [], [], options);
    toc();

    disp(norm(x_star - x_mega)/norm(x_mega));
end

% 100 dimensional problems
n = 100;

% -- random generated
for i = 1:10
    [Q, q, l, u, a, b, x_start] = generate_problem(n, 5);

    L = max(eig(Q));

    tic();
    [x_star, f_star, x_s, f_s, g_s] = KQP(Q, q, l, u, a, b , x_start, 10e-16, 10e-16, 1000, "fixed", 1/L, 0);
    toc();

    tic();
    x_mega = fmincon(@(x) objective_function(Q, q, x), x_start, [-eye(n); eye(n); -a'], [-l; u; -b], [], [], [], [], [], options);
    toc();

    disp(norm(x_star - x_mega)/norm(x_mega));
end