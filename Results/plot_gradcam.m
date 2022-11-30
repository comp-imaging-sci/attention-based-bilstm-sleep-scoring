[ha, pos] = tight_subplot(3, 3, [.03 .03],[.01 .01],[.01 .1]);
%wholebrain = [191030, 191115, 210108];
%hemis = [191030, 201125, 200115];
mouse = 191030;

load(sprintf('gradcam/%s_avg_hemis_label0.mat', num2str(mouse)));
a = ha(1);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), a, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label1.mat', num2str(mouse)));
b = ha(2);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), b, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label2.mat', num2str(mouse)));
c = ha(3);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), c, 0.5);

mouse = 201125;

load(sprintf('gradcam/%s_avg_hemis_label0.mat', num2str(mouse)));
a = ha(4);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), a, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label1.mat', num2str(mouse)));
b = ha(5);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), b, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label2.mat', num2str(mouse)));
c = ha(6);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), c, 0.5);

mouse = 200115;

load(sprintf('gradcam/%s_avg_hemis_label0.mat', num2str(mouse)));
a = ha(7);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), a, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label1.mat', num2str(mouse)));
b = ha(8);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), b, 0.5);

load(sprintf('gradcam/%s_avg_hemis_label2.mat', num2str(mouse)));
c = ha(9);
overlay_heatmap_gradcam(example_data(64,:,:), rescale(subject_avg_gradcam, 0, 1), c, 0.5);