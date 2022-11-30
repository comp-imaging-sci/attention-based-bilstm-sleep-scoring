function overlay_heatmap_gradcam(image, heatmap, base_axis, alpha)
set(base_axis, 'color', 'none', 'visible', 'off')
img = axes;
imagesc(squeeze(image)); 
colormap(img, 'gray'); 
set(img, 'color', 'none', 'visible', 'off')
hm = axes;
imagesc(heatmap, 'AlphaData', alpha);
colormap(hm, 'jet');
%colorbar
set(hm, 'color', 'none', 'visible', 'off')
linkprop([base_axis img hm], 'position');
end