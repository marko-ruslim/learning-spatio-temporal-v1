%% A demo of running LCA on a natural video with monophasic LGN
% The input to the model is pre-processed static natural images.
% This demo separates LGN input into ON and OFF channels.
% After learning, basis vectors should resembel Gabor like features.
% Note that a smaller video has been used in this demo
% Date: 20250320
% Author: Marko A. Ruslim

clear; clc; close all; tic;

%% LCA parameters

lca.lambda = 0.5; % The threshold of firing; determines the sparsity
lca.thresh_type = 'soft-non-negative';

lca.n_epoch = 300; % Number of epoches
lca.batch_size = 100; % Number of image patches in a minibatch
sz = 16; % Size of the video frame patch patch
L = sz^2; % Length of the input vector
OC = 1; % Overcompleteness
M = L; % Number of neurons in the network

lca.A_eta = 1e-1; % learning rate of A

history_flag = 1; % whether to display the history of response
lca.tau = 10; % ms
lca.dt = 1; % ms
lca_eta = lca.dt / lca.tau;
lca.n = 50; % number of iterations of computing stable response

%% Load spatially preprocessed video

toc; fprintf('Loading video.\n');
load('video_stim.mat'); toc;
num_frames = size(video, 3);
image_size_y = size(video, 1);
image_size_x = size(video, 2);
image_size = size(video,1); % Size of the images
BUFF = 4; % Buffer size that exclude image patches close to boundaries

%% Temporal LGN preprocessing

lgn.tau = 1; 
tmp = []; 
tau = lgn.tau;
for t = 0 : 10
    tmp(end+1) = (t^6*exp(-t/(tau/7)^(1/2)))/(720*(tau/7)^(7/2));
end

lgn.on_temp_filt_coef = tmp(end:-1:1)';
lgn.off_temp_filt_coef = lgn.on_temp_filt_coef; % monophasic

% LGN spatial preprocessing paramters
lgn.image_scale = 8; % Scale the input
lgn.sz_DoG = 16;
lgn.sigma_c = 1;
lgn.sigma_s = 1.5 * lgn.sigma_c;
lgn.sigma_d = lgn.sigma_s;

length_temp_filt = length(lgn.on_temp_filt_coef); 
% the length of the temporal window / the number of framed involved

X_spatiotemporal = zeros(L, length_temp_filt, lca.batch_size); 
% 256*4*100 matrix that stores frames in a temporal window

%% Weight and activation variables

lca.a_norm = 1;
lca.A = normalize_matrix(randn(2*L, M), 'L2 norm', lca.a_norm); 
% Connections between input and output layer

X = zeros(2*L, lca.batch_size); 
% Input matrix: each column is an image patch in the vector form (sz*sz, 1)

U = randn(M, lca.batch_size); 
% Membrane potentials of M neurons for lca.batch_size images

S = rand(M, lca.batch_size); 
% Firing rates (Response) of M neurons for lca.batch_size images

%% Display A and S

display_every = 100; % Frequency of generating plot
resize_factor = 3;
fig1 = figure(1);
display_matrix(lca.A,3); title('A'); colormap(gray);
fig11 = figure(11);
figure(2);
subplot(211);stem(S);
title(['S: the coefficients of ' num2str(lca.batch_size) 'patches']);

figure(100); ax_a = gca;
j = 0;
for i = 1 : size(lca.A, 1)
    ax_A{i} = animatedline(j, lca.A(i, 1), 'Color', rand([1,3]));
end
j = j+1;

%% Main loop: repeat the same video for lca.n_epoch times

for i_epoch = 1 : lca.n_epoch
    if i_epoch == 101
        lca.A_eta = 5e-2;
    end
    if i_epoch == 201
        lca.A_eta = 2e-2;
    end
    r_avg = 0;
    i_avg = 0;
    s_avg = 0;
    i_frame = 1;
    
    % Choose random some centers as a batch and keep these centers fixed 
    % for this epoch.
    r = BUFF + ceil((image_size-sz-2*BUFF)*rand(1, lca.batch_size));
    c = BUFF + ceil((image_size-sz-2*BUFF)*rand(1, lca.batch_size)); 
    
    % store the first few frames with the length of temporal window
    while i_frame <= length_temp_filt
        
        % Use batch training: in each batch of a iteration, there is 100
        % image patches with size sz*sz at 100 locations
        % For the first few frames, only store them into an array.
        for i_batch=1:lca.batch_size 
            % Shape the image patch into vector form (sz*sz, 1) 
            % where L = sz * sz
            X_data(:,i_batch) = reshape(video(r(i_batch):r(i_batch)+sz-...
                1,c(i_batch):c(i_batch)+sz-1,i_frame), L, 1);
            X_spatiotemporal(:, i_frame, i_batch) = X_data(:,i_batch);
        end
         fprintf('Epoch:%6d Iteration %6d\n', i_epoch, i_frame);
            
        i_frame = i_frame + 1;
    end
    
    fprintf('Epoch:%6d | Learning starts!\n', i_epoch);
    
    % Incorporate a periodic shift filter to cope with the incoming input
    filter_index = 1 : length_temp_filt;
    while i_frame <= num_frames
        
        % shift the component in the temporal filter
        filter_index = mod(filter_index-2,length_temp_filt)+1;
        
        for i_batch=1:lca.batch_size 
            % Shape the image patch into vector form (sz*sz, 1) 
            % where L = sz * sz
            X_data(:,i_batch) = reshape(video(r(i_batch):r(i_batch)+sz-...
                1,c(i_batch):c(i_batch)+sz-1,i_frame), L, 1);
            
            % Update the matrix that store past frames
            X_spatiotemporal(:, mod(i_frame-1,length_temp_filt)+1, ...
                i_batch) = X_data(:,i_batch);
            
            % temporal filtering for ON and OFF LGN channel
            X_data_ON(:,i_batch) = X_spatiotemporal(:,:,i_batch) * ...
                lgn.on_temp_filt_coef(filter_index); 
            X_data_OFF(:,i_batch) = X_spatiotemporal(:,:,i_batch) * ...
                lgn.off_temp_filt_coef(filter_index);  
        end
        
        % ON and OFF LGN input
        X_ON = max( X_data_ON, 0 );
        X_OFF = - min( X_data_OFF, 0 );
        X( 1:L, : ) = X_ON;
        X( L+1:2*L, : ) = X_OFF;
        
        % Compute the firing rate using LCA which implements sparse coding
        [S, U, S_his, U_his] = sparse_coding_by_LCA(...
            X, lca.A, lca.lambda, lca.thresh_type, lca_eta, lca.n, ...
            history_flag);
        
        R = X - lca.A * S; % Calculate residual error
        
        % Update bases
        dA = R * S' / lca.batch_size;
        lca.A = lca.A + lca.A_eta * dA;
        lca.A = normalize_matrix(lca.A, 'L2 norm', lca.a_norm); 
        % Normalize each column of the connection matrix
        
        % Save r_avg, s_avg, i_ave
        r_avg = r_avg + sum(sum(R.^2)) / L / lca.batch_size;
        i_avg = i_avg + sum(sum(X.^2)) / L / lca.batch_size;
        s_avg = s_avg + sum(S(:)~=0) / M / lca.batch_size;
        
        % Display connection matrix, A, and responses, S
        if (mod(i_frame,display_every) == 0)
            
            figure(1); % Display the connections from ON and OFF LGN cells 
            % to simple cells
            subplot 131; DisplayA( 'ONOFF', lca.A(1:2*L,:), resize_factor); 
            title('RFs: A_{ON}-A_{OFF}');
            subplot 132; DisplayA( 'ON', lca.A(1:2*L,:), resize_factor ); 
            title('A_{ON}');
            subplot 133; DisplayA( 'OFF', lca.A(1:2*L,:), resize_factor ); 
            title('A_{OFF}'); colormap(scm(256));

            figure(2); subplot(211);stem(S);
            title(['S: firing rates of ' num2str(lca.batch_size) ...
                ' patches']);
            subplot(212);stem(U);title(['U: membrane potentials of ' ...
                num2str(lca.batch_size) ' patches']);
            
            if history_flag == 1
                figure(3);
                plot(S_his); xlabel('Iterations of computing responses')
                title("Response trajectories of 1 example cell for " + ...
                    "input patches: check stability")
            end
            
            figure(4);
            display_matrix(X_data);title(['Input: ' ...
                num2str(lca.batch_size) ' input patches in the batch']);
            colorbar; colormap(gray);
            
            for i = 1 : size(lca.A, 1)
                addpoints(ax_A{i}, j, lca.A(i, 1));
            end
            j = j + 1;

            fprintf("Epoch:%6d Iteration %6d: Percentage of active " + ...
                "units: %3.1f%% | MSE: %3.1f%%\n", i_epoch, i_frame, ...
                100 * s_avg / i_frame, 100 * r_avg / i_avg);
            toc; drawnow limitrate;
        end
        i_frame = i_frame + 1;
    end
end