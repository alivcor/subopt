% Written By: alivcor (Abhinandan Dubey)
% Stony Brook University

%set the number of training examples here.
num_images = 10;
experiment_1_start = 1;
experiment_2_start =  11;
top_left = [0 0 224 224];
top_right = [289 0 224 224];
bottom_right = [289 289 224 224];
bottom_left = [0 289 224 224];
center = [146 146 223 223];
A = [];
net1 = load('vgg/imagenet-vgg-verydeep-16.mat') ;
net1 = vl_simplenn_tidy(net1) ;
vl_simplenn_display(net1) ;
image_count = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For experiment 1

for experiment_count=1:9
    current_dirname = num2str(experiment_count);
    dirname = strcat('dataset/', current_dirname);
    imagefiles = dir(strcat(dirname,'/*.png'));      
    nfiles = length(imagefiles);
    for ii=1:nfiles
        currentfilename = imagefiles(ii).name;
        X = ['Fetching Images', num2str(ii),' : ' , num2str(100*ii/nfiles), '% Complete.'];
        disp(X);
        im = single(imread(strcat(dirname, strcat('/', currentfilename))));
        im_(:,:,1) = im;
        im_(:,:,2) = im;
        im_(:,:,3) = im;
        im_1 = imcrop(im_, top_left); % TOP LEFT  [xmin ymin width height].
        im_2 = imcrop(im_, top_right); % TOP RIGHT
        im_3 = imcrop(im_, bottom_right); % BOTTOM RIGHT
        im_4 = imcrop(im_, bottom_left); % BOTTOM LEFT
        im_c = imcrop(im_, center); % CENTER
        imf_ = flip(im_,2);
        imf_1 = imcrop(imf_, top_left);
        imf_2 = imcrop(imf_, top_right);
        imf_3 = imcrop(imf_, bottom_right);
        imf_4 = imcrop(imf_, bottom_left);
        imf_c = imcrop(imf_, center);
        %im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ; % may help
        res1 = vl_simplenn(net1, im_1) ;
        res2 = vl_simplenn(net1, im_2) ;
        res3 = vl_simplenn(net1, im_3) ;
        res4 = vl_simplenn(net1, im_4) ;
        res5 = vl_simplenn(net1, im_c) ;
        res6 = vl_simplenn(net1, imf_1) ;
        res7 = vl_simplenn(net1, imf_2) ;
        res8 = vl_simplenn(net1, imf_3) ;
        res9 = vl_simplenn(net1, imf_4) ;
        res10 = vl_simplenn(net1, imf_c) ;
        repr1 = res1(36).x;
        repr2 = res2(36).x;
        repr3 = res3(36).x;
        repr4 = res4(36).x;
        repr5 = res5(36).x;
        repr6 = res6(36).x;
        repr7 = res7(36).x;
        repr8 = res8(36).x;
        repr9 = res9(36).x;
        repr10 = res10(36).x;
        vgg_features = squeeze((repr1+repr2+repr3+repr4+repr5+repr6+repr7+repr8+repr9+repr10)/10);
        feature_vector = [image_count vgg_features' experiment_count];
        A = cat(1,A,feature_vector);
        image_count = image_count + 1;
    end
end

image_count = image_count - 1;
alldata = A;

links = [];
[p, q] = meshgrid(1:image_count, 1:image_count);
mask   = triu(ones(image_count), 1) > 0.5;
pairs  = [p(mask) q(mask)];

for i=1:size(pairs,1)
    im1_exp = floor(pairs(i,1)/21)+1;
    im2_exp = floor(pairs(i,2)/21)+1;
    if im1_exp == im2_exp
        same_exp = 1;
    else
        same_exp = 0;
    end
    links = cat(1,links,[pairs(i,1) pairs(i,2) same_exp]);
 end 

disp(size(links));

%Save the learned feature vectors to a file
save('saved/alldata.mat', 'alldata');
save('saved/links.mat', 'links');
