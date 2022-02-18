bunch_file_name = "bunch_300.mat";
max_iters = 500;

load(bunch_file_name)
k = length(bunch_cel);

timing_quadprog = zeros(1, k);
f_limit_quadprog = zeros(1, k);


% fixed step size
x_s_kqp_mean_fs = zeros(1, max_iters+1);
f_s_kqp_mean_fs = zeros(1, max_iters+1);

x_limit_fixed = zeros(1, k);
f_limit_fixed = zeros(1, k);
timing_kqp_fs = zeros(1, k);

% diminishing
x_s_kqp_mean_diminishing = zeros(1, max_iters+1);
f_s_kqp_mean_diminishing = zeros(1, max_iters+1);

x_limit_diminishing = zeros(1, k);
f_limit_diminishing = zeros(1, k);
timing_kqp_diminishing = zeros(1, k);

% polyak
x_s_kqp_mean_polyak = zeros(1, max_iters+1);
f_s_kqp_mean_polyak = zeros(1, max_iters+1);

x_limit_polyak = zeros(1, k);
f_limit_polyak = zeros(1, k);
timing_kqp_polyak = zeros(1, k);

% armijo_i
x_s_kqp_mean_armijo_i = zeros(1, max_iters+1);
f_s_kqp_mean_armijo_i = zeros(1, max_iters+1);

x_limit_armijo_i = zeros(1, k);
f_limit_armijo_i = zeros(1, k);
timing_kqp_armijo_i = zeros(1, k);

% armijo_ii
x_s_kqp_mean_armijo_ii = zeros(1, max_iters+1);
f_s_kqp_mean_armijo_ii = zeros(1, max_iters+1);

x_limit_armijo_ii = zeros(1, k);
f_limit_armijo_ii = zeros(1, k);
timing_kqp_armijo_ii = zeros(1, k);

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
    
    f = @(x) objective_function(Q,q,x);
    
    % FIXED ---------
    
    tic;
    [~, ~, x_s_kqp_fs, f_s_kqp_fs, g_s_kqp_fs] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "fixed", 1/L, 0, 0);
    timing_kqp_fs(i) = toc;
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_fs - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_fs - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_fs = x_s_kqp_mean_fs + x_seq_padded; 
    f_s_kqp_mean_fs = f_s_kqp_mean_fs + f_seq_padded;
    
    x_limit_fixed(i) = x_seq_padded(end);
    f_limit_fixed(i) = f_seq_padded(end);
    
    % DIMINISHING ---------

    tic;
    [~, ~, x_s_kqp_diminishing, f_s_kqp_diminishing, g_s_kqp_diminishing] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "diminishing", @(i) 1/(L*i),0, 0);
    timing_kqp_diminishing(i) = toc;

    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_diminishing - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_diminishing - f_star)/norm(f_star), max_iters);
    
    x_s_kqp_mean_diminishing = x_s_kqp_mean_diminishing + x_seq_padded;
    f_s_kqp_mean_diminishing = f_s_kqp_mean_diminishing + f_seq_padded;
    
    x_limit_diminishing(i) = x_seq_padded(end);
    f_limit_diminishing(i) = f_seq_padded(end);
    
    % POLYAK ---------
    
    tic;
    [~, ~, x_s_kqp_polyak, f_s_kqp_polyak, g_s_kqp_polyak] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "polyak", @(i) L^2/i, 0, 0);
    timing_kqp_polyak(i) = toc;
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_polyak - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_polyak - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_polyak = x_s_kqp_mean_polyak + x_seq_padded;
    f_s_kqp_mean_polyak = f_s_kqp_mean_polyak + f_seq_padded;
    
    x_limit_polyak(i) = x_seq_padded(end);
    f_limit_polyak(i) = f_seq_padded(end);
    
    % ARMIJO I ---------
    
    tic;
    [~, ~, x_s_kqp_armijo_i, f_s_kqp_armijo_i, g_s_kqp_armijo_i] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "armijo", {0.5, 0.1}, 0, 0);
    timing_kqp_armijo_i(i) = toc;        
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_armijo_i - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_armijo_i - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_armijo_i = x_s_kqp_mean_armijo_i + x_seq_padded;
    f_s_kqp_mean_armijo_i = f_s_kqp_mean_armijo_i + f_seq_padded;
    
    x_limit_armijo_i(i) = x_seq_padded(end);
    f_limit_armijo_i(i) = f_seq_padded(end);
    
    % ARMIJO II ---------
    
    tic;
    [~, ~, x_s_kqp_armijo_ii, f_s_kqp_armijo_ii, g_s_kqp_armijo_ii] = KQP(f, l, u, a, b , x_start, 1e-6, 1e-15, max_iters, "armijo_ii", {0.5, 0.1}, 0, 0);
    timing_kqp_armijo_ii(i) = toc;        
    
    x_seq_padded = padding_sequence(vecnorm(x_s_kqp_armijo_ii - x_star)/norm(x_star), max_iters);
    f_seq_padded = padding_sequence(abs(f_s_kqp_armijo_ii - f_star)/abs(f_star), max_iters);
    
    x_s_kqp_mean_armijo_ii = x_s_kqp_mean_armijo_ii + x_seq_padded;
    f_s_kqp_mean_armijo_ii = f_s_kqp_mean_armijo_ii + f_seq_padded;
    
    x_limit_armijo_ii(i) = x_seq_padded(end);
    f_limit_armijo_ii(i) = f_seq_padded(end);

    % FMINCON ---------
        
    tic;
    [~, f_star_quadprog] = minimize_matlab_kqp(x_start, Q, q, l, u, a, b, max_iters, false);
    timing_quadprog(i) = toc;
    f_limit_quadprog(i) = abs(f_star_quadprog - f_star)/abs(f_star);
    
    % ---------
    
    wait_bar = waitbar(i/k, wait_bar,'Processing your data');
end

% average over sequences of convergence

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

fprintf("convergence time quadprog step size, mean %d, std %d\n", mean(timing_fmincon), std(timing_fmincon));
fprintf("relative error on the f reached quadprog step size, mean %d, var %d\n\n", mean(f_limit_quadprog), var(f_limit_quadprog));

% plot the convergence curve

% semilogy(x_s_kqp_mean_fs);
% hold on
% semilogy(x_s_kqp_mean_diminishing);
% semilogy(x_s_kqp_mean_polyak);
% semilogy(x_s_kqp_mean_armijo_i);
% semilogy(x_s_kqp_mean_armijo_ii);
% semilogy(x_s_fmincon_mean);
% legend('fixed','diminishing', 'polyak', 'armijo_i', 'armijo_ii', 'fmincon');
% title('Relative norm of xs to x*');
% hold off

figure('DefaultAxesFontSize',18);
semilogy(remove_padding_from_sequence(f_s_kqp_mean_fs), 'LineWidth',1);
hold on
semilogy(remove_padding_from_sequence(f_s_kqp_mean_diminishing), 'LineWidth',1);
semilogy(remove_padding_from_sequence(f_s_kqp_mean_polyak), 'LineWidth',1);
semilogy(remove_padding_from_sequence(f_s_kqp_mean_armijo_i), 'LineWidth',1);
semilogy(remove_padding_from_sequence(f_s_kqp_mean_armijo_ii), 'LineWidth',1);
lgd = legend('fixed','diminishing', 'polyak', 'armijo 1', 'armijo 2');
lgd.FontSize = 15;

title('Relative norm of fs to f*', 'FontSize', 18);
xlim([0, 501]);
hold off


function [seq] = padding_sequence(sequence, max_iter)
    size_sequence = size(sequence);
    filled = size_sequence(2);
    
    padding_size = max_iter - filled + 1;
    
    padding = repmat(sequence(:, end), 1, padding_size);
    
    seq = [sequence, padding];
end


function [seq] = remove_padding_from_sequence(sequence)
    min_seq = min(sequence);
    seq = sequence(sequence > min_seq + 1e-15);
    seq = [seq, min_seq];
end