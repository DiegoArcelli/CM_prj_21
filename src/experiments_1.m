bunch_file_name = "bunch.mat";
max_iters = 100;

clear global x_s_fmincon;
clear global f_s_fmincon;

global x_s_fmincon;
global f_s_fmincon;

x_s_fmincon_mean = zeros(1, max_iters+1);
f_s_fmincon_mean = zeros(1, max_iters+1);

% fixed step size
x_s_kqp_mean_fs = zeros(1, max_iters+1);
f_s_kqp_mean_fs = zeros(1, max_iters+1);

% diminishing
x_s_kqp_mean_diminishing = zeros(1, max_iters+1);
f_s_kqp_mean_diminishing = zeros(1, max_iters+1);

% polyak
x_s_kqp_mean_polyak = zeros(1, max_iters+1);
f_s_kqp_mean_polyak = zeros(1, max_iters+1);

% armijo
x_s_kqp_mean_armijo = zeros(1, max_iters+1);
f_s_kqp_mean_armijo = zeros(1, max_iters+1);

load(bunch_file_name)

k = length(bunch_cel);

% bunch_cel is the name of the bunch of problems
for problem = bunch_cel
    
    p = problem{1};
    
    Q = p.Q;
    q = p.q;
    l = p.l;
    u = p.u;
    a = p.a;
    b = p.b;
    x_start = p.x_start;
    x_star = p.x_star;
    f_star = p.f_star;
    
    % --- repeat this for every step size to test
    tic();
    L = max(eig(Q));
    [~, ~, x_s_kqp_fs, f_s_kqp_fs, g_s_kqp_fs] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-6, max_iters, "fixed", 1/L, 0);
    toc();
    
    x_s_kqp_mean_fs = x_s_kqp_mean_fs + padding_sequence(vecnorm(x_s_kqp_fs - x_star)/norm(x_star), max_iters);
    f_s_kqp_mean_fs = f_s_kqp_mean_fs + padding_sequence(abs(f_s_kqp_fs - f_star)/abs(f_star), max_iters);

    tic();
    [~, ~, x_s_kqp_diminishing, f_s_kqp_diminishing, g_s_kqp_diminishing] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-6, max_iters, "diminishing", @(i) i/L, 0);
    toc();
    
    x_s_kqp_mean_diminishing = x_s_kqp_mean_diminishing + padding_sequence(vecnorm(x_s_kqp_diminishing - x_star)/norm(x_star), max_iters);
    f_s_kqp_mean_diminishing = f_s_kqp_mean_diminishing + padding_sequence(abs(f_s_kqp_diminishing - f_star)/abs(f_star), max_iters);

    tic();
    [~, ~, x_s_kqp_polyak, f_s_kqp_polyak, g_s_kqp_polyak] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-6, max_iters, "polyak", 0.1, 0);
    toc();
    
    x_s_kqp_mean_polyak = x_s_kqp_mean_polyak + padding_sequence(vecnorm(x_s_kqp_polyak - x_star)/norm(x_star), max_iters);
    f_s_kqp_mean_polyak = f_s_kqp_mean_polyak + padding_sequence(abs(f_s_kqp_polyak - f_star)/abs(f_star), max_iters);

    tic();
    [~, ~, x_s_kqp_armijo, f_s_kqp_armijo, g_s_kqp_armijo] = KQP(Q, q, l, u, a, b , x_start, 1e-6, 1e-6, max_iters, "armijo", {0.5, 0.5, 0.1}, 0);
    toc();
        
    x_s_kqp_mean_armijo = x_s_kqp_mean_armijo + padding_sequence(vecnorm(x_s_kqp_armijo - x_star)/norm(x_star), max_iters);
    f_s_kqp_mean_armijo = f_s_kqp_mean_armijo + padding_sequence(abs(f_s_kqp_armijo - f_star)/abs(f_star), max_iters);

    x_s_fmincon = [];
    f_s_fmincon = [];
    
    tic();
    minimize_matlab_kqp(x_start, Q, q, l, u, a, b, max_iters, false);
    toc();
    
    x_s_fmincon_mean =  x_s_fmincon_mean + padding_sequence(vecnorm(x_s_fmincon - x_star)/norm(x_star), max_iters);
    f_s_fmincon_mean =  f_s_fmincon_mean + padding_sequence(abs(f_s_fmincon - f_star)/abs(f_star), max_iters);
end

x_s_kqp_mean_fs =  x_s_kqp_mean_fs / k;
f_s_kqp_mean_fs =  f_s_kqp_mean_fs / k;

x_s_kqp_mean_diminishing =  x_s_kqp_mean_diminishing / k;
f_s_kqp_mean_diminishing =  f_s_kqp_mean_diminishing / k;

x_s_kqp_mean_polyak =  x_s_kqp_mean_polyak / k;
f_s_kqp_mean_polyak =  f_s_kqp_mean_polyak / k;

x_s_kqp_mean_armijo =  x_s_kqp_mean_armijo / k;
f_s_kqp_mean_armijo =  f_s_kqp_mean_armijo / k;

x_s_fmincon_mean =  x_s_fmincon_mean / k;
f_s_fmincon_mean =  f_s_fmincon_mean / k;

% plot the convergence curve

semilogy(x_s_kqp_mean_fs);
hold on
semilogy(x_s_kqp_mean_diminishing);
semilogy(x_s_kqp_mean_polyak);
semilogy(x_s_kqp_mean_armijo);
semilogy(x_s_fmincon_mean);
hold off

input("");

semilogy(f_s_kqp_mean_fs);
hold on
semilogy(f_s_kqp_mean_diminishing);
semilogy(f_s_kqp_mean_polyak);
semilogy(f_s_kqp_mean_armijo);
semilogy(f_s_fmincon_mean);
hold off


function [seq] = padding_sequence(sequence, max_iter)
    % docstring
    size_sequence = size(sequence);
    filled = size_sequence(2);
    
    padding_size = max_iter - filled + 1;
    
    padding = repmat(sequence(:, end), 1, padding_size);
    
    seq = [sequence, padding];
end