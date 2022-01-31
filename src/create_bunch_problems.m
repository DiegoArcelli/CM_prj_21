n = 200;
scale = 10;
n_samples = 100;
file_name = "bunch.mat";

bunch_cel = cell(1, n_samples);

wait_bar = waitbar(0,'Creating samples');

for i = 1:n_samples
    [Q, q, l, u, a, b, x_start] = generate_problem(n, scale);
    
    [x_star, f_star] = minimize_matlab_kqp(x_start, Q, q, l, u, a, b, -1, true);
    
    problem.Q = Q;
    problem.q = q;
    problem.l = l;
    problem.u = u;
    problem.a = a;
    problem.b = b;
    problem.x_start = x_start;
    problem.x_star = x_star;
    problem.f_star = f_star;
    
    bunch_cel{i} = problem;
    
    wait_bar = waitbar(i/n_samples, wait_bar,'Creating samples');
end

save(file_name, "bunch_cel");