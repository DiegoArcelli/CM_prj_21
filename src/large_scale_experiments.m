n = 1000;
scale = 10;
n_samples = 1;
inter_per  = 1;
actv_per = -1;
max_iters = 500;

% fixed step size
timing_kqp_fs = zeros(1, n_samples);
relative_error_kqp_fs = zeros(1, n_samples);

% polyak
timing_kqp_polyak = zeros(1, n_samples);
relative_error_kqp_polyak = zeros(1, n_samples);

i = 0;

wait_bar = waitbar(0,'Processing your data');

% bunch_cel is the name of the bunch of problems
for problem_instance = 1:n_samples
    i = i+1;

    [Q, q, l, u, a, b, x_start] = generate_problem(n, scale, inter_per, actv_per);
        
    
    tic;
    [x_star, f_star] = minimize_matlab_kqp(x_start, Q, q, l, u, a, b, -1, true);
    toc;
    
    tic;
    eigs_Q = eigs(Q, 1);
    tic;
    
    L = max(eigs_Q);

    f = @(x) objective_function(Q,q,x);
    
    % FIXED ---------
    
    tic;
    [~, f_fixed, ~, ~, ~] = KQP(f, l, u, a, b , x_start, 1e-15, 1e-15, max_iters, "fixed", 1/L, 0, 0);
    timing_kqp_fs(i) = toc;
    relative_error_kqp_fs(i) = abs(f_fixed - f_star)/abs(f_star);
    
    disp(relative_error_kqp_fs);
    
    % POLYAK ---------
    
    tic;
    [~, f_polyak, ~, ~, ~] = KQP(f, l, u, a, b , x_start, 1e-15, 1e-15, max_iters, "polyak", @(i) L^2/i, 0, 0);
    timing_kqp_polyak(i) = toc;
    relative_error_kqp_polyak(i) = abs(f_polyak - f_star)/abs(f_star);
    
    % ---------
    
    wait_bar = waitbar(i/n_samples, wait_bar,'Processing your data');
end

% display the statistics over execution timing

fprintf("convergence time fixed step size, mean %d, std %d\n", mean(timing_kqp_fs), std(timing_kqp_fs));
fprintf("relative error fixed step size, mean %d, std %d\n", mean(relative_error_kqp_fs), std(relative_error_kqp_fs));

fprintf("convergence time polyak step size, mean %d, std %d\n", mean(timing_kqp_polyak), std(timing_kqp_polyak));
fprintf("relative error fixed step size, mean %d, std %d\n", mean(relative_error_kqp_polyak), std(relative_error_kqp_polyak));