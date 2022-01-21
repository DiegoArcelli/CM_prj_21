training_dataset = readtable('monks-1-train.csv');
training_dataset = table2array(training_dataset(:,1:end-1));

X_tr = onehotencode(training_dataset(:, 2:end));
d_tr = training_dataset(:, 1);

disp(size(X_tr));

d_tr(d_tr==0)=-1;

testing_dataset = readtable('monks-1-test.csv');
testing_dataset = table2array(testing_dataset(:,1:end-1));

X_ts = testing_dataset(:, 2:end);
d_ts = testing_dataset(:, 1);

d_ts(d_ts==0)=-1;

f = @(alpha) f_svm(alpha, d_tr.*X_tr);

C = 100;

N = size(X_tr);
N = N(1);

l = zeros([N, 1]);
u = repmat(C, [N, 1]);

x_start = zeros([N, 1]);

% [alpha_star, f_star, x_s, f_s, g_s] = KQP(f, l, u, d_tr, 0, x_start, 10e-15, 10e-15, 1000, "fixed", 0.1, 0, 1);

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'active-set', 'OptimalityTolerance', 10e-15, 'ConstraintTolerance', 10e-15, 'MaxIterations', 1000, 'FunctionTolerance', 10e-16);
alpha_star = fmincon(f, x_start, [-eye(N); eye(N)], [-l; u], d_tr', 0, [], [], [], options);

svs_idx = alpha_star ~= 0;
alpha_gez = alpha_star(svs_idx);
d_svs = d_tr(svs_idx);
X_svs = X_tr(svs_idx, :);


% poco elegante ma funzionale come direbbe mio nonno
on_margin_sv_idx = 0 < alpha_star & alpha_star < C;
on_margin_sv = X_tr(on_margin_sv_idx, :);
on_margin_sv = on_margin_sv(1,:);
on_margin_d = d_tr(on_margin_sv_idx);
on_margin_d = on_margin_d(1);

b_o = 0;

b_o = b_o + on_margin_d - (alpha_gez'*(d_svs.*X_svs))*on_margin_sv';

preds = sign((alpha_gez'*(d_svs.*X_svs))*X_ts' + b_o);
preds = preds';

%disp(preds);

disp(1 - mean(abs(d_ts - preds)/2));

preds = sign((alpha_gez'*(d_svs.*X_svs))*X_tr' + b_o);
preds = preds';

%disp(preds);

disp(1 - mean(abs(d_tr - preds)/2));


