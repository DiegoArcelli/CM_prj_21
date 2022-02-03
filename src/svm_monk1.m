load fisheriris.mat

% transform to binary classification problem
idx_bc = species ~= "setosa";
X = meas(idx_bc,:);
species_binary = species(idx_bc);
d = zeros(100, 1);
d(species_binary=="versicolor") = 1;
d(species_binary=="virginica") = -1;

random_permutation = randperm(100);

X = X(random_permutation, :);
d = d(random_permutation);

X_tr = X(1:90, :);
d_tr = d(1:90);

X_ts = X(91:end, :);
d_ts = d(91:end);

SVMModel = fitcsvm(X_tr,d_tr, 'BoxConstraint',10);

preds = predict(SVMModel,X_tr);

disp(1 - mean(abs(d_tr - preds)/2));


f = @(alpha) f_svm(alpha, d_tr.*X_tr);

C = 10;

N = size(X_tr);
N = N(1);

l = zeros([N, 1]);
u = repmat(C, [N, 1]);

x_start = zeros([N, 1]);

[alpha_star, f_star, x_s, f_s, g_s] = KQP(f, l, u, d_tr, 0, x_start, 1e-15, 1e-15, 1000, "diminishing", @(i) 1/i, 0, 1);

% options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'active-set', 'OptimalityTolerance', 10e-15, 'ConstraintTolerance', 10e-15, 'MaxIterations', 1000, 'FunctionTolerance', 10e-16);
% alpha_star = fmincon(f, x_start, [-eye(N); eye(N)], [-l; u], d_tr', 0, [], [], [], options);

svs_idx = alpha_star ~= 0;
alpha_gez = alpha_star(svs_idx);
d_svs = d_tr(svs_idx);
X_svs = X_tr(svs_idx, :);


on_margin_sv_idx = 0 < alpha_star & alpha_star < C;
on_margin_sv = X_tr(on_margin_sv_idx, :);
% on_margin_sv = on_margin_sv(1,:);
on_margin_d = d_tr(on_margin_sv_idx);
% on_margin_d = on_margin_d(1);

b_o = 0;

disp(length(on_margin_sv));
disp(size(on_margin_sv));

% add the mean over the on margin vectors to have numerical stability
N_prime = size(on_margin_sv);
N_prime = N_prime(1);
for i= 1:N_prime
    sv = on_margin_sv(i,:);
    d_sv = on_margin_d(i);
    b_o = b_o + (d_sv - (alpha_gez'*(d_svs.*X_svs))*sv');
end

b_o = b_o/N_prime;

% preds = sign((alpha_gez'*(d_svs.*X_svs))*X_ts' + b_o);
% preds = preds';

%disp(preds);

% disp(1 - mean(abs(d_ts - preds)/2));

preds = sign((alpha_gez'*(d_svs.*X_svs))*X_tr' + b_o);
preds = preds';

%disp(preds);

disp(1 - mean(abs(d_tr - preds)/2));


