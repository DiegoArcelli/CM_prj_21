bunch_file_name = "bunch.mat";
max_iters = 500;

clear global x_s_fmincon;
clear global f_s_fmincon;

global x_s_fmincon;
global f_s_fmincon;

x_s_fmincon_mean = zeros(1, max_iters+1);
f_s_fmincon_mean = zeros(1, max_iters+1);
x_limit_fmincon = zeros(1, max_iters);
f_limit_fmincon = zeros(1, max_iters);
timing_fmincon = zeros(1, max_iters);

% fixed step size
x_s_kqp_mean_fs = zeros(1, max_iters+1);
f_s_kqp_mean_fs = zeros(1, max_iters+1);
x_limit_fixed = zeros(1, max_iters);
f_limit_fixed = zeros(1, max_iters);
timing_kqp_fs = zeros(1, max_iters);

% diminishing
x_s_kqp_mean_diminishing = zeros(1, max_iters+1);
f_s_kqp_mean_diminishing = zeros(1, max_iters+1);
x_limit_diminishing = zeros(1, max_iters);
f_limit_diminishing = zeros(1, max_iters);
timing_kqp_diminishing = zeros(1, max_iters);

% polyak
x_s_kqp_mean_polyak = zeros(1, max_iters+1);
f_s_kqp_mean_polyak = zeros(1, max_iters+1);
x_limit_polyak = zeros(1, max_iters);
f_limit_polyak = zeros(1, max_iters);
timing_kqp_polyak = zeros(1, max_iters);

% armijo_i
x_s_kqp_mean_armijo_i = zeros(1, max_iters+1);
f_s_kqp_mean_armijo_i = zeros(1, max_iters+1);
x_limit_armijo_i = zeros(1, max_iters);
f_limit_armijo_i = zeros(1, max_iters);
timing_kqp_armijo_i = zeros(1, max_iters);

% armijo_ii
x_s_kqp_mean_armijo_ii = zeros(1, max_iters+1);
f_s_kqp_mean_armijo_ii = zeros(1, max_iters+1);
x_limit_armijo_ii = zeros(1, max_iters);
f_limit_armijo_ii = zeros(1, max_iters);
timing_kqp_armijo_ii = zeros(1, max_iters);

load(bunch_file_name)

k = length(bunch_cel);
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
    x_star = p.x_star;
    f_star = p.f_star;
    
    eigs_Q = eig(Q);
    L = max(eigs_Q);
    tau = min(eigs_Q);

    tic;
    [~, ~, x_s_kqp_fs, f_s_kqp_fs, g_s_kqp_fs] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "fixed", 1/L, 0);
    timing_kqp_fs(i) = toc;
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_fs - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_fs - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_fs = x_s_kqp_mean_fs + x_seq_padded; 
    f_s_kqp_mean_fs = f_s_kqp_mean_fs + f_seq_padded;
    
    x_limit_fixed(i) = x_seq_padded(end);
    f_limit_fixed(i) = f_seq_padded(end);

    tic;
    [~, ~, x_s_kqp_diminishing, f_s_kqp_diminishing, g_s_kqp_diminishing] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "diminishing", @(i) 1/(L*i), 0);
    timing_kqp_diminishing(i) = toc;

    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_diminishing - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(vecnorm(x_s_kqp_diminishing - x_star)/norm(x_star), max_iters);
    
    x_s_kqp_mean_diminishing = x_s_kqp_mean_diminishing + x_seq_padded;
    f_s_kqp_mean_diminishing = f_s_kqp_mean_diminishing + f_seq_padded;
    
    x_limit_diminishing(i) = x_seq_padded(end);
    f_limit_diminishing(i) = f_seq_padded(end);

    tic;
    [~, ~, x_s_kqp_polyak, f_s_kqp_polyak, g_s_kqp_polyak] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "polyak", @(i) L^2/i, 0);
    timing_kqp_polyak(i) = toc;
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_polyak - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_polyak - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_polyak = x_s_kqp_mean_polyak + x_seq_padded;
    f_s_kqp_mean_polyak = f_s_kqp_mean_polyak + f_seq_padded;
    
    x_limit_polyak(i) = x_seq_padded(end);
    f_limit_polyak(i) = f_seq_padded(end);
    
    tic;
    [~, ~, x_s_kqp_armijo_i, f_s_kqp_armijo_i, g_s_kqp_armijo_i] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "armijo", {0.5, 0.01}, 0);
    timing_kqp_armijo_i(i) = toc;        
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_armijo_i - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_armijo_i - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_armijo_i = x_s_kqp_mean_armijo_i + x_seq_padded;
    f_s_kqp_mean_armijo_i = f_s_kqp_mean_armijo_i + f_seq_padded;
    
    x_limit_armijo_i(i) = x_seq_padded(end);
    f_limit_armijo_i(i) = f_seq_padded(end);
    
    tic;
    [~, ~, x_s_kqp_armijo_ii, f_s_kqp_armijo_ii, g_s_kqp_armijo_ii] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "armijo_ii", {0.5, 0.5}, 0);
    timing_kqp_armijo_ii(i) = toc;        
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_armijo_ii - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_armijo_ii - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_armijo_ii = x_s_kqp_mean_armijo_ii + x_seq_padded;
    f_s_kqp_mean_armijo_ii = f_s_kqp_mean_armijo_ii + f_seq_padded;
    
    x_limit_armijo_ii(i) = x_seq_padded(end);
    f_limit_armijo_ii(i) = f_seq_padded(end);

    x_s_fmincon = [];
    f_s_fmincon = [];
    
    tic;
    minimize_matlab_kqp(x_start, Q, q, l, u, a, b, max_iters, false);
    timing_fmincon(i) = toc;
    
    x_seq_padded = padding_sequence(vecnorm(x_s_fmincon - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_fmincon - f_star)/abs(f_star), max_iters);
    
    x_s_fmincon_mean =  x_s_fmincon_mean + x_seq_padded;
    f_s_fmincon_mean =  f_s_fmincon_mean + f_seq_padded;
    
    x_limit_fmincon(i) = x_seq_padded(end);
    f_limit_fmincon(i) = f_seq_padded(end);
    
    wait_bar = waitbar(i/k, wait_bar,'Processing your data');
end

x_s_kqp_mean_fs =  x_s_kqp_mean_fs / k;
f_s_kqp_mean_fs =  f_s_kqp_mean_fs / k;

x_s_kqp_mean_diminishing =  x_s_kqp_mean_diminishing / k;
f_s_kqp_mean_diminishing =  f_s_kqp_mean_diminishing / k;

x_s_kqp_mean_polyak =  x_s_kqp_mean_polyak / k;
f_s_kqp_mean_polyak =  f_s_kqp_mean_polyak / k;

x_s_kqp_mean_armijo_i =  x_s_kqp_mean_armijo_i / k;
f_s_kqp_mean_armijo_i =  f_s_kqp_mean_armijo_i / k;

x_s_kqp_mean_armijo_ii =  x_s_kqp_mean_armijo_ii / k;
f_s_kqp_mean_armijo_ii =  f_s_kqp_mean_armijo_ii / k;

x_s_fmincon_mean =  x_s_fmincon_mean / k;
f_s_fmincon_mean =  f_s_fmincon_mean / k;

% display the statistics over execution

fprintf("convergence time fixed step size, mean %d, std %d\n", mean(timing_kqp_fs), std(timing_kqp_fs));
fprintf("relative error on the x reached fixed step size, mean %d, var %d\n", mean(x_limit_fixed), var(x_limit_fixed));
fprintf("relative error on the f reached fixed step size, mean %d, var %d\n\n", mean(f_limit_fixed), var(f_limit_fixed));

fprintf("convergence time diminishing step size, mean %d, std %d\n", mean(timing_kqp_diminishing), std(timing_kqp_diminishing));
fprintf("relative error on the x reached diminishing step size, mean %d, var %d\n", mean(x_limit_diminishing), var(x_limit_diminishing));
fprintf("relative error on the f reached diminishing step size, mean %d, var %d\n\n", mean(f_limit_diminishing), var(f_limit_diminishing));

fprintf("convergence time polyak step size, mean %d, std %d\n", mean(timing_kqp_polyak), std(timing_kqp_polyak));
fprintf("relative error on the x reached polyak step size, mean %d, var %d\n", mean(x_limit_polyak), var(x_limit_polyak));
fprintf("relative error on the f reached polyak step size, mean %d, var %d\n\n", mean(f_limit_polyak), var(f_limit_polyak));

fprintf("convergence time armijo_i step size, mean %d, std %d\n", mean(timing_kqp_armijo_i), std(timing_kqp_armijo_i));
fprintf("relative error on the x reached armijo_i step size, mean %d, var %d\n", mean(x_limit_armijo_i), var(x_limit_armijo_i));
fprintf("relative error on the f reached armijo_i step size, mean %d, var %d\n\n", mean(f_limit_armijo_i), var(f_limit_armijo_i));

fprintf("convergence time armijo_ii step size, mean %d, std %d\n", mean(timing_kqp_armijo_ii), std(timing_kqp_armijo_ii));
fprintf("relative error on the x reached armijo_ii step size, mean %d, var %d\n", mean(x_limit_armijo_ii), var(x_limit_armijo_ii));
fprintf("relative error on the f reached armijo_ii step size, mean %d, var %d\n\n", mean(f_limit_armijo_ii), var(f_limit_armijo_ii));

fprintf("convergence time fmincon step size, mean %d, std %d\n", mean(timing_fmincon), std(timing_fmincon));
fprintf("relative error on the x reached fmincon step size, mean %d, var %d\n", mean(x_limit_fmincon), var(x_limit_fmincon));
fprintf("relative error on the f reached fmincon step size, mean %d, var %d\n\n", mean(f_limit_fmincon), var(f_limit_fmincon));

% plot the convergence curve

semilogy(x_s_kqp_mean_fs);
hold on
semilogy(x_s_kqp_mean_diminishing);
semilogy(x_s_kqp_mean_polyak);
semilogy(x_s_kqp_mean_armijo_i);
semilogy(x_s_kqp_mean_armijo_ii);
semilogy(x_s_fmincon_mean);
legend('fixed','diminishing', 'polyak', 'armijo_i', 'armijo_ii', 'fmincon');
title('Relative norm of xs to x*');
hold off

input("");

semilogy(f_s_kqp_mean_fs);
hold on
semilogy(f_s_kqp_mean_diminishing);
semilogy(f_s_kqp_mean_polyak);
semilogy(f_s_kqp_mean_armijo_i);
semilogy(f_s_kqp_mean_armijo_ii);
semilogy(f_s_fmincon_mean);
legend('fixed','diminishing', 'polyak', 'armijo_i', 'armijo_ii', 'fmincon');
title('Relative norm of fs to f*');
hold off


function [seq] = padding_sequence(sequence, max_iter)
    % docstring
    size_sequence = size(sequence);
    filled = size_sequence(2);
    
    padding_size = max_iter - filled + 1;
    
    padding = repmat(sequence(:, end), 1, padding_size);
    
    seq = [sequence, padding];
end