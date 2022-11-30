load 200128_10s.mat;
load 200128_attention_weights.mat;

%% Visualize attention weights color coded by the sleep states
NREMepochs=find(y_preds==1);
REMepochs=find(y_preds==2);
Wakeepochs=find(y_preds==0);

for i = 1:size(att_weights, 1)
    att_weights(i, :) = rescale(att_weights(i, :), 0, 1);
end
%att_weights = reshape(att_weights, size(att_weights, 1) * size(att_weights, 2), 1);


%%
figure;
subplot(131);bar(mean(att_weights(Wakeepochs, :), 1), 1, 'k'); ylim([0, 0.15]); title('Wakefulness');
ylabel('Average attention weights')
xlabel('Time steps')
sum(mean(att_weights(Wakeepochs, :), 1), 'all')
set(gca, 'FontSize', 14)

subplot(132);bar(mean(att_weights(NREMepochs, :), 1), 1, 'b'); ylim([0, 0.15]); title('NREM');
xlabel('Time steps')
sum(mean(att_weights(NREMepochs, :), 1), 'all')
set(gca, 'FontSize', 14)

subplot(133);bar(mean(att_weights(REMepochs, :), 1), 1, 'r'); ylim([0, 0.15]); title('REM');
xlabel('Time steps')
sum(mean(att_weights(REMepochs, :), 1), 'all')
set(gca, 'FontSize', 14)
%% Visualize attention weights color coded by the sleep states

b = bar(att_weights, 10);
ylim([0 1])

b.FaceColor = 'flat';
for z=1:length(NREMepochs)
   %b.CData((NREMepochs(z)-1)*168+1:NREMepochs(z)*168,:) = repmat([39 125 229]/255, 168, 1);
   b.CData((NREMepochs(z)-1)*168+1:NREMepochs(z)*168,:) = repmat([0 0 0], 168, 1);
end

for z=1:length(Wakeepochs)
   %b.CData((Wakeepochs(z)-1)*168+1:Wakeepochs(z)*168,:) = repmat([241 161 5]/255, 168, 1);
    b.CData((Wakeepochs(z)-1)*168+1:Wakeepochs(z)*168,:) = repmat([0 0 1], 168, 1);
end

for z=1:length(REMepochs)
   %b.CData((REMepochs(z)-1)*168+1:REMepochs(z)*168,:) = repmat([38 179 150]/255, 168, 1);
   b.CData((REMepochs(z)-1)*168+1:REMepochs(z)*168,:) = repmat([1 0 0], 168, 1);
end

xlabel('Time (hr)','FontWeight', 'bold')
ylabel('Attention weights')
set(gca,'XTick',[180 360 540 720 900 1060]*168);
set(gca,'XTickLabel', [0.5 1.0 1.5 2.0 2.5 3.0],'Fontsize',14);
set(gca, 'FontSize', 16, 'FontWeight', 'bold');

%% visualize attention weights in a histogram
Wake_att_weights = zeros(168*numel(Wakeepochs), 1);
for z=1:length(Wakeepochs)
   Wake_att_weights((z-1)*168+1:z*168) =att_weights((Wakeepochs(z)-1)*168+1:Wakeepochs(z)*168);
end
subplot(131); histogram(Wake_att_weights, 100, 'Normalization', 'probability'); title('Wakefulness');

NREM_att_weights = zeros(168*numel(NREMepochs), 1);
for z=1:length(NREMepochs)
   NREM_att_weights((z-1)*168+1:z*168) =att_weights((NREMepochs(z)-1)*168+1:NREMepochs(z)*168);
end
subplot(132); histogram(NREM_att_weights, 100,  'Normalization', 'probability');title('NREM');

REM_att_weights = zeros(168*numel(REMepochs), 1);
for z=1:length(REMepochs)
   REM_att_weights((z-1)*168+1:z*168) =att_weights((REMepochs(z)-1)*168+1:REMepochs(z)*168);
end
subplot(133); histogram(REM_att_weights, 100,  'Normalization', 'probability'); title('REM');



%%
deletes = [];
trans_weights = [];
gap = 30;
for i=2:numel(y_trues)
    if y_trues(i) ~= y_trues(i-1)
        trans_weights = cat(1, trans_weights, att_weights((i-1)*168-gap:(i-1)*168+gap));
        deletes = cat(2, deletes, (i-1)*168-gap:(i-1)*168+gap);
    end
end
att_weights(deletes) = [];
histogram(trans_weights, 100, 'Normalization', 'probability'); hold on;
histogram(att_weights, 100, 'Normalization', 'probability');