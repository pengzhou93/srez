f_lst = dir(fullfile('291', '*'));
mkdir('291_jpg');
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile('291',f_info.name);
    %disp(f_path);
    img_raw = imread(f_path);
%     else
%         img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
   
    
    %img_raw = im2double(img_raw); 
    %img_size = size(img_raw);
    %img_raw=imresize(img_raw,[256,256]);

    
    %img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    %img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    %img_4 = imresize(img_raw,1/4,'bicubic');
    
    %patch_name = sprintf('%s/%d',folder,count);
    
%     save(patch_name, 'img_raw');
%     save(sprintf('%s_2', patch_name), 'img_2');
%     save(sprintf('%s_3', patch_name), 'img_3');
%     save(sprintf('%s_4', patch_name), 'img_4');
    path = fullfile('291_jpg',[f_info.name(1:end-3),'jpg']);
    imwrite(img_raw,path)
    %path2=fullfile('test',[f_info.name(1:end-3),'jpg']);
     %imwrite(img_4,path2)
    %count = count + 1;
    %display(count);
    
    
end
