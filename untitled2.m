
y_lst = dir(fullfile('residual_gan/test_set14', '*.png'));
f_lst = dir(fullfile('residual_gan/Set14_train', '*.jpg'));
p = 0;
for f_iter = 1:numel(f_lst)
  f_info = f_lst(f_iter);
  y_info = y_lst(f_iter);
  f_path = fullfile('residual_gan/Set14_train',f_info.name);
  img_raw = imread(f_path);
  r = img_raw;
  if size(img_raw,3)>1
     img_raw = rgb2ycbcr(img_raw);
     y_path = fullfile('residual_gan/test_set14',['subpixel_',num2str(f_iter-1),'.png']);
     img =imread(y_path);
     img_raw(:,:,1) = img; 
     img_raw = ycbcr2rgb(img_raw);
  else
      y_path = fullfile('residual_gan/test_set14',['subpixel_',num2str(f_iter-1),'.png']);
      img2 =imread(y_path);
      img_raw = img2;
  end
  new_path = fullfile('residual_gan/test_set14_yuv',f_info.name);
  imwrite(img_raw,new_path);
 p = p+psnr(img_raw,r)
end
p = p/f_iter