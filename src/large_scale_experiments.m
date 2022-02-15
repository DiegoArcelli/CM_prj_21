n = 10000;
scale = 10;
n_samples = 10;
inter_per  = 1;
actv_per = -1;
max_iters = 500;

file_name = "bunch.mat";

bunch_cel = cell(1, n_samples);

wait_bar = waitbar(0,'Creating samples');

for i = 1:n_samples
    [Q, q, l, u, a, b, x_start] = generate_problem(n, scale, inter_per, actv_per);
       
    problem.Q = Q;
    problem.q = q;
    problem.l = l;
    problem.u = u;
    problem.a = a;
    problem.b = b;
    problem.x_start = x_start;

    bunch_cel{i} = problem;
    
    wait_bar = waitbar(i/n_samples, wait_bar,'Creating samples');
end

% fixed step size
timing_kqp_fs = zeros(1, n_samples);

% polyak
timing_kqp_polyak = zeros(1, n_samples);

i = 0;

wait_bar = waitbar(0,'Processing your data');

% bunch_cel is the name of the bunch of problems
for problem_instance = bunch_cel
    i = i+1;
    
    p = problem_instance{1};
    
    Q = p.Q;
    q = p.q;
    l = p.l;
    u = p.u;
    a = p.a;
    b = p.b;
    x_start = p.x_start;
    
    eigs_Q = eig(Q);
    L = max(eigs_Q);
    tau = min(eigs_Q);

    f = @(x) objective_function(Q,q,x);
    
    % FIXED ---------
    
    tic;
    [~, ~, x_s_kqp_fs, f_s_kqp_fs, g_s_kqp_fs] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "fixed", 1/L, 0, 0);
    timing_kqp_fs(i) = toc;
    
    % POLYAK ---------
    
    tic;
    [~, ~, x_s_kqp_polyak, f_s_kqp_polyak, g_s_kqp_polyak] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "polyak", @(i) L^2/i, 0, 0);
    timing_kqp_polyak(i) = toc;
    
    % ---------
    
    wait_bar = waitbar(i/n_samples, wait_bar,'Processing your data');
end

% display the statistics over execution timing

fprintf("convergence time fixed step size, mean %d, std %d\n", mean(timing_kqp_fs), std(timing_kqp_fs));

fprintf("convergence time polyak step size, mean %d, std %d\n", mean(timing_kqp_polyak), std(timing_kqp_polyak));