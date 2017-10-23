count = 0;
% f_lst = dir(fullfile('set142', '*'));
% for f_iter = 1:numel(f_lst)
% %     disp(f_iter);
%     f_info = f_lst(f_iter);
%     if f_info.name == '.'
%         continue;
%     end
%     f_path = fullfile('set142',f_info.name);
    %disp(f_path);
    f_path = fullfile('Set142','comic.jpg')
    img_raw = imread(f_path);
    img_raw = padarray(img_raw,[23,6],'replicate','post');
    path = fullfile('splitimage','comic.jpg')
%     imwrite(img_raw,path);
% end
img_size = size(img_raw);
patch_size = 64;
stride = 64;
x_size = (img_size(2)-patch_size)/stride+1;
y_size = (img_size(1)-patch_size)/stride+1;
for x = 0:x_size-1
        for y = 0:y_size-1
            count = count+1;
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf(['comic',num2str(count),'.jpg']);
            patch = (img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:));
            path2 = fullfile('splitimage',patch_name);
            imwrite(patch, path2);
        end
end