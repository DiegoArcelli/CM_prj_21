% ----- LOAD DATASET FOR BINARY CLASSIFICATION
% X, Y
load ionosphere.mat

d = zeros(size(Y));

d(contains(Y, 'b')) = 1;
d(contains(Y, 'g')) = -1;

N = size(X); N = N(1);
random_permutation = randperm(N);

X = X(random_permutation, :);
d = d(random_permutation);

% taking 80% for training
Tr_size = floor((N/100)*80);

X_tr = X(1:Tr_size, :);
d_tr = d(1:Tr_size);

X_ts = X(Tr_size+1:end, :);
d_ts = d(Tr_size+1:end);

% -----------

% C parameter
C = 10;

% matlab solver
tic();
SVMModel = fitcsvm(X_tr,d_tr, 'BoxConstraint',C);
timinig_matlab_svm = toc();

tr_acc_matlab_svm = compute_accuracy(SVMModel, X_tr, d_tr);
ts_acc_matlab_svm = compute_accuracy(SVMModel, X_ts, d_ts);

fprintf("Matlab svm solver: timinig = %d\n", timinig_matlab_svm);
fprintf("Matlab svm solver: training accuracy = %0.2f\n", tr_acc_matlab_svm);
fprintf("Matlab svm solver: test accuracy = %0.2f\n\n", ts_acc_matlab_svm);

% fixed step size personal solver
tic();
fixed_svm = KQPSVM(C, "fixed", 0.0001);
fixed_svm = fixed_svm.fit(X_tr, d_tr);
timinig_fixed = toc();

tr_acc_fixed = compute_accuracy(fixed_svm, X_tr, d_tr);
ts_acc_fixed = compute_accuracy(fixed_svm, X_ts, d_ts);

fprintf("Fixed step size svm solver: timinig = %d\n", timinig_fixed);
fprintf("Fixed step size svm solver: training accuracy = %0.2f\n", tr_acc_fixed);
fprintf("Fixed step size svm solver: test accuracy = %0.2f\n\n", ts_acc_fixed);

% diminishing step size personal solver
tic();
diminishing_svm = KQPSVM(C, "diminishing", @(i) 1/i);

diminishing_svm = diminishing_svm.fit(X_tr, d_tr);
timinig_diminishing = toc();

tr_acc_diminishing = compute_accuracy(diminishing_svm, X_tr, d_tr);
ts_acc_diminishing = compute_accuracy(diminishing_svm, X_ts, d_ts);

fprintf("Diminishing step size svm solver: timinig = %d\n", timinig_diminishing);
fprintf("Diminishing step size svm solver: training accuracy = %0.2f\n", tr_acc_diminishing);
fprintf("Diminishing step size svm solver: test accuracy = %0.2f\n\n", ts_acc_diminishing);

% polyak step size personal solver
tic();
polyak_svm = KQPSVM(C, "polyak", @(i) 1/i);

polyak_svm = polyak_svm.fit(X_tr, d_tr);
timinig_polyak = toc();

tr_acc_polyak = compute_accuracy(polyak_svm, X_tr, d_tr);
ts_acc_polyak = compute_accuracy(polyak_svm, X_ts, d_ts);

fprintf("Polyak step size svm solver: timinig = %d\n", timinig_polyak);
fprintf("Polyak step size svm solver: training accuracy = %0.2f\n", tr_acc_polyak);
fprintf("Polyak step size svm solver: test accuracy = %0.2f\n\n", ts_acc_polyak);

% armijo step size personal solver
tic();
armijo_svm = KQPSVM(C, "armijo", {0.5, 0.1});

armijo_svm = armijo_svm.fit(X_tr, d_tr);
timinig_armijo = toc();

tr_acc_armijo = compute_accuracy(armijo_svm, X_tr, d_tr);
ts_acc_armijo = compute_accuracy(armijo_svm, X_ts, d_ts);

fprintf("Armijo step size svm solver: timinig = %d\n", timinig_armijo);
fprintf("Armijo step size svm solver: training accuracy = %0.2f\n", tr_acc_armijo);
fprintf("Armijo step size svm solver: test accuracy = %0.2f\n\n", ts_acc_armijo);


% armijo 2 step size personal solver
tic();
armijo_svm_ii = KQPSVM(C, "armijo_ii", {0.5, 0.1});

armijo_svm_ii = armijo_svm_ii.fit(X_tr, d_tr);
timinig_armijo_ii = toc();

tr_acc_armijo_ii = compute_accuracy(armijo_svm_ii, X_tr, d_tr);
ts_acc_armijo_ii = compute_accuracy(armijo_svm_ii, X_ts, d_ts);

fprintf("Armijo 2 step size svm solver: timinig = %d\n", timinig_armijo_ii);
fprintf("Armijo 2 step size svm solver: training accuracy = %0.2f\n", tr_acc_armijo_ii);
fprintf("Armijo 2 step size svm solver: test accuracy = %0.2f\n\n", ts_acc_armijo_ii);



function [accuracy] = compute_accuracy(model, X, d)
    preds = predict(model, X);

    accuracy = 1 - mean(abs(d - preds)/2);
end

