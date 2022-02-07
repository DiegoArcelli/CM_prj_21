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

Tr_size = floor((N/100)*80);

X_tr = X(1:Tr_size, :);
d_tr = d(1:Tr_size);

X_ts = X(Tr_size+1:end, :);
d_ts = d(Tr_size+1:end);

% -----------

% C parameter
C = 10;

tic();
SVMModel = fitcsvm(X_tr,d_tr, 'BoxConstraint',C);
toc();

disp(compute_accuracy(SVMModel, X_tr, d_tr));

disp(compute_accuracy(SVMModel, X_ts, d_ts));

tic();
mysvm = KQPSVM(C, "diminishing", @(i) 1/i);

mysvm = mysvm.fit(X_tr, d_tr);
toc();

disp(compute_accuracy(mysvm, X_tr, d_tr));

disp(compute_accuracy(mysvm, X_ts, d_ts));

tic();
mysvm = KQPSVM(C, "polyak", @(i) 1/i);

mysvm = mysvm.fit(X_tr, d_tr);
toc();

disp(compute_accuracy(mysvm, X_tr, d_tr));

disp(compute_accuracy(mysvm, X_ts, d_ts));

tic();
mysvm = KQPSVM(C, "armijo", {0.5, 0.1});

mysvm = mysvm.fit(X_tr, d_tr);
toc();

disp(compute_accuracy(mysvm, X_tr, d_tr));

disp(compute_accuracy(mysvm, X_ts, d_ts));


function [accuracy] = compute_accuracy(model, X, d)
    preds = predict(model, X);

    accuracy = 1 - mean(abs(d - preds)/2);
end

