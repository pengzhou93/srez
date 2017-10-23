
f_lst = dir(fullfile('residual_gan/Set142', '*.jpg'));
for f_iter = 1:numel(f_lst)
  f_info = f_lst(f_iter);
  f_path = fullfile('residual_gan/Set142',f_info.name);
  img_raw = imread(f_path);
  height = size(img_raw,1);
  new_height = fix(height/8)*8;
  width = size(img_raw,2);
  new_width = fix(width/8)*8;
  img_raw = imcrop(img_raw,[0,0, new_width,new_height]);
  new_path = fullfile('residual_gan/Set14_train',f_info.name);
  imwrite(img_raw,new_path);
end


% a=a(1:360,1:248,:);
% b=b(1:656,1:528,:);
% imwrite(a,'/home/machlearn/gengcong/residual_gan/Set142/comic2.jpg');
% imwrite(b,'/home/machlearn/gengcong/residual_gan/Set142/ppt2.jpg');
