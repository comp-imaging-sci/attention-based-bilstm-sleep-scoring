function plot_hypnogram(y_trues, y_preds1, y_preds2)
%simple function to plot the hypnogram alone based on the scoringindex
disp('Plotting hypnogram');

timemax=length(y_trues);
figure;
subplot(3,1,1);
axis ([0 timemax+20 0 3])
set(gca,'YTick',[0.5 1.5 2.5])
set(gca,'YTickLabel',{'Wakefulness','NREM','REM'},'FontWeight', 'bold')
set(gca,'XTick',[180 360 540 720 900 1060]);
set(gca,'XTickLabel', [0.5 1.0 1.5 2.0 2.5 3.0],'Fontsize',14);
%xlabel('TIme (hr)','FontWeight', 'bold')
title('WFCI-based CNN-LSTM classification');
set(gcf,'color','white')
set(gca, 'FontSize', 16);
for l=1:numel(y_trues)
    switch y_trues(l)
        
        case 0  %WAKE
            line([1 1]*l,[0 1],'LineWidth',1,'Color',[241 161 5]/255);
            
        case 1 %NREM
            line([1 1]*l,[1 2],'LineWidth',1,'Color',[39 125 229]/255);

        case 2 %REM
            line([1 1]*l,[2 3],'LineWidth',1,'Color',[38 179 150]/255);
    end
end

subplot(3,1,2);
axis ([0 timemax+20 0 3])
set(gca,'YTick',[0.5 1.5 2.5])
set(gca,'YTickLabel',{'Wakefulness','NREM','REM'},'FontWeight', 'bold')
set(gca,'XTick',[180 360 540 720 900 1060]);
set(gca,'XTickLabel', [0.5 1.0 1.5 2.0 2.5 3.0],'Fontsize',14);
%xlabel('Time (hr)','FontWeight', 'bold')
title('WFCI-based MVG-CNN classification');
set(gcf,'color','white')
set(gca, 'FontSize', 16);
for l=1:numel(y_preds1)
    switch y_preds1(l)
        
        case 0  %WAKE
            line([1 1]*l,[0 1],'LineWidth',1,'Color',[241 161 5]/255);
            
        case 1 %NREM
            line([1 1]*l,[1 2],'LineWidth',1,'Color',[39 125 229]/255);

        case 2 %REM
            line([1 1]*l,[2 3],'LineWidth',1,'Color',[38 179 150]/255);
    end
end

subplot(3,1,3);
axis ([0 timemax+20 0 3])
set(gca,'YTick',[0.5 1.5 2.5])
set(gca,'YTickLabel',{'Wakefulness','NREM','REM'},'FontWeight', 'bold')
set(gca,'XTick',[180 360 540 720 900 1060]);
set(gca,'XTickLabel', [0.5 1.0 1.5 2.0 2.5 3.0],'Fontsize',14);
xlabel('Time (hr)','FontWeight', 'bold')
title('EEG/EMG-based human scoring');
set(gcf,'color','white')
set(gca, 'FontSize', 16);
for l=1:numel(y_preds2)
    switch y_preds2(l)
        
        case 0  %WAKE
            line([1 1]*l,[0 1],'LineWidth',1,'Color',[241 161 5]/255);
            
        case 1 %NREM
            line([1 1]*l,[1 2],'LineWidth',1,'Color',[39 125 229]/255);

        case 2 %REM
            line([1 1]*l,[2 3],'LineWidth',1,'Color',[38 179 150]/255);
    end
end
disp('Finished plotting hypnogram');
end