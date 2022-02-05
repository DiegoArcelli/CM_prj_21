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

disp(SVMModel);

preds = predict(SVMModel,X_tr);

disp(1 - mean(abs(d_tr - preds)/2));

mysvm = KQPSVM(10, "diminishing", @(i) 1/i);

mysvm = mysvm.fit(X_tr, d_tr);

preds = mysvm.predict(X_tr);

disp(1 - mean(abs(d_tr - preds)/2));

