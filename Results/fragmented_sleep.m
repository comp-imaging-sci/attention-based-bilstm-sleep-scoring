%% Fragmented sleep (state duration and number of state transition)
clc; clear all;
load 200128_mvgcnn_10s.mat;
y_preds_mvgcnn = y_preds;
load 200128_10s.mat;
plot_hypnogram(y_preds, y_preds_mvgcnn, y_trues);

load 200128_nick_eric_scoring.mat
plot_hypnogram_two_raters(y_trues_eric, y_trues_nick);

%% State duration
% human scoring
scoring = y_trues;
epoch_duration = 10;
% WAKE
wake_duration = {};
num_epoch = 0;
chunk_idx = 0;
for i=1:length(scoring)
    
    if scoring(i) == 0
        num_epoch = num_epoch+1;
        if num_epoch == 1
            chunk_idx = chunk_idx+1;
        end
        wake_duration{chunk_idx} = num_epoch;
    else
        num_epoch = 0;        
    end
            
end

% NREM
nrem_duration = {};
num_epoch = 0;
chunk_idx = 0;
for i=1:length(scoring)
    
    if scoring(i) == 1
        num_epoch = num_epoch+1;
        if num_epoch == 1
            chunk_idx = chunk_idx+1;
        end
        nrem_duration{chunk_idx} = num_epoch;
    else
        num_epoch = 0;        
    end
            
end

% REM
rem_duration = {};
num_epoch = 0;
chunk_idx = 0;
for i=1:length(scoring)
    
    if scoring(i) == 2
        num_epoch = num_epoch+1;
        if num_epoch == 1
            chunk_idx = chunk_idx+1;
        end
        rem_duration{chunk_idx} = num_epoch;
    else
        num_epoch = 0;        
    end
            
end
clc;
% compute average state duration 
fprintf('WAKE: %.f s\nNREM: %.f s\nREM: %.f s\n', mean(cell2mat(wake_duration))*epoch_duration, mean(cell2mat(nrem_duration))*epoch_duration, mean(cell2mat(rem_duration))*epoch_duration);


%% State transition
num_tran = 0;
for i=1:length(y_trues)-1
    if scoring(i) ~= scoring(i+1)
        num_tran = num_tran + 1;       
    end
end
fprintf('Number of state transitions: %d\n', num_tran);