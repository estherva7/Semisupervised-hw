load mnist_uint8.mat
X = [train_x; test_x];
X = double(X)/255; %Normalize 0-1
y_vec = [train_y; test_y];
y = vec2ind(y_vec');
Nl = round(plabeled/100*numel(y));
    
Xl = X(1:Nl,:); Xu = X(Nl+1:end,:);
yl = y(1:Nl); yu = y(Nl+1:end);
yl_vec = y_vec(1:Nl,:); yu_vec= y_vec(Nl+1,:);
num_labels = 10;

layers = [784 100]; %946   199   101   100

disp(['Network size: ' num2str(layers)]);

lambda = 0;
noise = 0.5;
max_iter = 1;

% Pretrain model using stacked denoising auto-encoders

no_layers = length(layers);
model = cell(2 * no_layers, 1);

for i=1:no_layers
    [network, mappedX] = train_autoencoder(X, layers(i), noise, max_iter,lambda);
    model{i}.W        = network{1}.W;
    model{i}.bias_upW = network{1}.bias_upW;
end

for i=1:no_layers
    model{no_layers + i}.W        = model{no_layers - i + 1}.W';
    if i ~= no_layers
        model{no_layers + i}.bias_upW = model{no_layers - i}.bias_upW;
    else
        model{no_layers + i}.bias_upW = zeros(1, size(X, 2));
    end
end
clear network mappedX
 
% Compute mean squared error of initial model predictions
reconX = run_data_through_autoenc(model, X);
disp(['MSE of initial model: ' num2str(mean((reconX(:) - X(:)) .^ 2))]);

% Finetune model using gradient descent
noise = 0.5;
max_iter = 20;
lambda = 0;

model = backprop(model, X, X, max_iter, noise, lambda);

% Compute mean squared error of final model predictions
[reconX, mappedXtrain] = run_data_through_autoenc(model, Xl);
disp(['MSE of final model: ' num2str(mean((reconX(:) - Xl(:)) .^ 2))]);

[reconX, mappedXtest] = run_data_through_autoenc(model, X);
disp(['MSE of final model: ' num2str(mean((reconX(:) - X(:)) .^ 2))]);


%%

modelSVM = logregFit(mappedXtrain, yl);
[yhat] = logregPredict(modelSVM, mappedXtest);      

% %%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
% 
% %  Setup and train a stacked denoising autoencoder (SDAE)
% 
% rand('state',0)
% 
% sae = saesetup([784 200 150 100]);
% 
% sae.ae{1}.activation_function       = 'sigm';
% 
% sae.ae{1}.learningRate              = 1;
% 
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
% 
% opts.numepochs =   10;
% 
% opts.batchsize = 1;
% 
% sae = saetrain(sae, Xtrain, opts);
% 
% visualize(sae.ae{1}.W{1}(:,2:end)')
% 
%  
% 
% % Use the SDAE to initialize a FFNN
% 
% nn = nnsetup([784 100 10]);
% 
% nn.activation_function              = 'sigm';
% 
% nn.learningRate                     = 1;
% 
% nn.W{1} = sae.ae{1}.W{1};
% 
%  
% 
% % Train the FFNN
% 
% opts.numepochs =   20;
% 
% opts.batchsize = 100;
% 
% nn = nntrain(nn, Xtrain, test_y, opts);
% 
% [er, bad] = nntest(nn, Xtest, train_y);
% 
% er
% 
% assert(er < 0.16, 'Too big error');
% 
% % Use another classifier instead
% Xtrain_feat = Xtrain*sae.ae{1,1}.W{1,2};
% Xtest_feat = Xtest*sae.ae{1,1}.W{1,2};
% 
% modelSVM = logregFit(Xtrain_feat, ytrain);
% [yhat] = logregPredict(modelSVM, Xtest_feat);  
% 


