%set the number of training examples here.
numtr = 3000;
numtest_from = numtr + 1;
numtest_to_s2 = 3564;
numtest_to_s3 = 3468;

%Get all data (thread-id, label) for Season 2
[thread_ids_s2, thread_lbs_s2] = textread('data/Dataset/Label/BBTS02_label_number_new', ...
'thread%d %d', 'headerlines', 1);
data_s2 = struct('threadIds_s2', thread_ids_s2, 'threadlbs_s2', thread_lbs_s2);
assignin('base', 'data_s2', data_s2);           
%Save the data into a file
save('data/data_s2.mat', 'data_s2');

%Get all data (thread-id, label) for Season 3
[thread_ids_s3, thread_lbs_s3] = textread('data/Dataset/Label/BBTS03_label_number.txt', ...
'thread%d %d', 'headerlines', 1);
data_s3 = struct('threadIds_s3', thread_ids_s3, 'threadlbs_s3', thread_lbs_s3);
assignin('base', 'data_s3', data_s3);           
%Save the data into a file
save('data/data_s3.mat', 'data_s3');

%we plan to use first 3000 for training
A = [];
net1 = load('cnn/src/imagenet-vgg-verydeep-16.mat') ;
net1 = vl_simplenn_tidy(net1) ;
vl_simplenn_display(net1) ;


%get VGG Reps for all Season 2 thread images
for n = 1:numtr
    tempA = [];
    fname = 'data/Dataset/BBT_S02/thread';
    fnum = num2str(n);
    fnumsize = size(fnum, 2);
    ttr = 4 - fnumsize;
    for k = 1:ttr
        fnum = strcat('0', fnum);
    end
    fname = strcat(fname, fnum);
    X = ['Fetching Thread', fnum,' : ' , num2str(100*n/numtr), '% Complete.'];
    disp(X);
    fname = strcat(fname, '/');
    %get a list of all jpg files
    fnsyn = strcat(fname, '*.jpg');
    %disp(fname);
    %disp(fnsyn);
    imagefiles = dir(fnsyn);
    nfiles = length(imagefiles);
    ofname = fname;
    for ii=1:nfiles
        fname = ofname;
        currentfilename = imagefiles(ii).name;
        disp(currentfilename);
        fname = strcat(fname, currentfilename);
        im = single(imread(fname));
        kimsize = size(im);
        kimsize = min(kimsize(1), kimsize(2));
        im_ = imresize(im,(256/kimsize));
        im_1 = imcrop(im_, [1 1 223 223]);
        im_2 = imcrop(im_, [231 1 223 223]);
        im_3 = imcrop(im_, [231 33 223 223]);
        im_4 = imcrop(im_, [0 33 224 224]);
        im_c = imcrop(im_, [116 17 223 223]);
        imf_ = flip(im_,2);
        imf_1 = imcrop(imf_, [1 1 223 223]);
        imf_2 = imcrop(imf_, [231 1 223 223]);
        imf_3 = imcrop(imf_, [231 33 223 223]);
        imf_4 = imcrop(imf_, [0 33 224 224]);
        imf_c = imcrop(imf_, [116 17 223 223]);
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
        fresult = (repr1+repr2+repr3+repr4+repr5+repr6+repr7+repr8+repr9+repr10)/10;
        tempA = cat(4,tempA,fresult);
    end
    nReps = size(tempA,4);
    finA = mean(tempA,4);
%     disp(size(fresult));
%     disp(size(finA));
    A = cat(4,A,finA);
end





%get VGG Reps for all Season 3 thread images
for n = 1:numtr
    tempA = [];
    fname = 'data/Dataset/BBT_S03/thread';
    fnum = num2str(n);
    fnumsize = size(fnum, 2);
    ttr = 4 - fnumsize;
    for k = 1:ttr
        fnum = strcat('0', fnum);
    end
    fname = strcat(fname, fnum);
    X = ['Fetching Thread', fnum,' : ' , num2str(100*n/numtr), '% Complete.'];
    disp(X);
    fname = strcat(fname, '/');
    %get a list of all jpg files
    fnsyn = strcat(fname, '*.jpg');
    %disp(fname);
    %disp(fnsyn);
    imagefiles = dir(fnsyn);
    nfiles = length(imagefiles);
    ofname = fname;
    for ii=1:nfiles
        fname = ofname;
        currentfilename = imagefiles(ii).name;
        disp(currentfilename);
        fname = strcat(fname, currentfilename);
        im = single(imread(fname));
        kimsize = size(im);
        kimsize = min(kimsize(1), kimsize(2));
        im_ = imresize(im,(256/kimsize));
        im_1 = imcrop(im_, [1 1 223 223]);
        im_2 = imcrop(im_, [231 1 223 223]);
        im_3 = imcrop(im_, [231 33 223 223]);
        im_4 = imcrop(im_, [0 33 224 224]);
        im_c = imcrop(im_, [116 17 223 223]);
        imf_ = flip(im_,2);
        imf_1 = imcrop(imf_, [1 1 223 223]);
        imf_2 = imcrop(imf_, [231 1 223 223]);
        imf_3 = imcrop(imf_, [231 33 223 223]);
        imf_4 = imcrop(imf_, [0 33 224 224]);
        imf_c = imcrop(imf_, [116 17 223 223]);
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
        fresult = (repr1+repr2+repr3+repr4+repr5+repr6+repr7+repr8+repr9+repr10)/10;
        tempA = cat(4,tempA,fresult);
    end
    nReps = size(tempA,4);
    finA = mean(tempA,4);
%     disp(size(fresult));
%     disp(size(finA));
    A = cat(4,A,finA);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alldata_s2 = load('data/data_s2.mat') ;
alldata_s3 = load('data/data_s3.mat') ;

a1_s2 = transpose(alldata_s2.data_s2.threadIds_s2);
a1_s3 = transpose(alldata_s3.data_s3.threadIds_s3);

b1_s2 = a1_s2(1,1:numtr);
b1_s3 = a1_s3(1,1:numtr);

b1 = cat(2, b1_s2, b1_s3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a2_s2 = transpose(alldata_s2.data_s2.threadlbs_s2);
a2_s3 = transpose(alldata_s3.data_s3.threadlbs_s3);

b2_s2 = a2_s2(1,1:numtr);
b2_s3 = a2_s3(1,1:numtr);

b2 = cat(2, b2_s2, b2_s3);

b3 = ones(1,2*numtr);

images = struct('id', b1, 'data', A, 'label', b2, 'set', b3);
assignin('base', 'images', images);
            
%Save the learned feature vectors to a file
save('data/fvsdb.mat', 'images');


% Load the learned feature vectors file
imdb = load('data/fvsdb.mat') ;

net = initializeBBTCNN() ;

% train and evaluate the CNN 

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 15 ;
trainOpts.continue = true ;
opts.train.gpus = [] ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'myresults' ;

% Convert to a GPU array if needed
if opts.train.gpus == [1]
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if opts.train.gpus == [1]
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
save('myresults/bbtcnn.mat', '-struct', 'net') ;

% APPLY THE MODEL
net = load('myresults/bbtcnn.mat') ;

net1 = load('cnn/src/imagenet-vgg-verydeep-16.mat') ;
net1 = vl_simplenn_tidy(net1) ;

alldata_s2 = load('data/data_s2.mat');
alldata_s3 = load('data/data_s3.mat');
idnames = cat(1, alldata_s2.data_s2.threadIds_s2(numtest_from:numtest_to_s2), alldata_s3.data_s3.threadIds_s3(numtest_from:numtest_to_s3));




testlabels = [];
for n = numtest_from:numtest_to_s2
    tempA = [];
    fname = 'data/Dataset/BBT_S02/thread';
    fnum = num2str(n);
    fnumsize = size(fnum, 2);
    ttr = 4 - fnumsize;
    for k = 1:ttr
        fnum = strcat('0', fnum);
    end
    fname = strcat(fname, fnum);
    X = ['Fetching Thread', fnum,' : ' , num2str(100*(n-numtest_from)/(numtest_to_s2-numtest_from)), '% Complete.'];
    disp(X);
    fname = strcat(fname, '/');
    %get a list of all jpg files
    fnsyn = strcat(fname, '*.jpg');
    %disp(fname);
    %disp(fnsyn);
    imagefiles = dir(fnsyn);
    nfiles = length(imagefiles);
    ofname = fname;
    for ii=1:nfiles
        fname = ofname;
        currentfilename = imagefiles(ii).name;
        disp(currentfilename);
        fname = strcat(fname, currentfilename);
        im = single(imread(fname));
        kimsize = size(im);
        kimsize = min(kimsize(1), kimsize(2));
        im_ = imresize(im,(256/kimsize));
        im_1 = imcrop(im_, [1 1 223 223]);
        im_2 = imcrop(im_, [231 1 223 223]);
        im_3 = imcrop(im_, [231 33 223 223]);
        im_4 = imcrop(im_, [0 33 224 224]);
        im_c = imcrop(im_, [116 17 223 223]);
        imf_ = flip(im_,2);
        imf_1 = imcrop(imf_, [1 1 223 223]);
        imf_2 = imcrop(imf_, [231 1 223 223]);
        imf_3 = imcrop(imf_, [231 33 223 223]);
        imf_4 = imcrop(imf_, [0 33 224 224]);
        imf_c = imcrop(imf_, [116 17 223 223]);
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
        fresult = (repr1+repr2+repr3+repr4+repr5+repr6+repr7+repr8+repr9+repr10)/10;
        tempA = cat(4,tempA,fresult);
    end
    nReps = size(tempA,4);
    finA = mean(tempA,4);
%     disp(size(fresult));
%     disp(size(finA));
    res = vl_simplenn(net, finA) ;
    clbl = res(2).x;
    j = find(clbl==max(clbl));
    disp(j)
    testlabels = cat(1,testlabels,j);
end


for n = numtest_from:numtest_to_s3
    tempA = [];
    fname = 'data/Dataset/BBT_S03/thread';
    fnum = num2str(n);
    fnumsize = size(fnum, 2);
    ttr = 4 - fnumsize;
    for k = 1:ttr
        fnum = strcat('0', fnum);
    end
    fname = strcat(fname, fnum);
    X = ['Fetching Thread', fnum,' : ' , num2str(100*(n-numtest_from)/(numtest_to_s3-numtest_from)), '% Complete.'];
    disp(X);
    fname = strcat(fname, '/');
    %get a list of all jpg files
    fnsyn = strcat(fname, '*.jpg');
    %disp(fname);
    %disp(fnsyn);
    imagefiles = dir(fnsyn);
    nfiles = length(imagefiles);
    ofname = fname;
    for ii=1:nfiles
        fname = ofname;
        currentfilename = imagefiles(ii).name;
        disp(currentfilename);
        fname = strcat(fname, currentfilename);
        im = single(imread(fname));
        kimsize = size(im);
        kimsize = min(kimsize(1), kimsize(2));
        im_ = imresize(im,(256/kimsize));
        im_1 = imcrop(im_, [1 1 223 223]);
        im_2 = imcrop(im_, [231 1 223 223]);
        im_3 = imcrop(im_, [231 33 223 223]);
        im_4 = imcrop(im_, [0 33 224 224]);
        im_c = imcrop(im_, [116 17 223 223]);
        imf_ = flip(im_,2);
        imf_1 = imcrop(imf_, [1 1 223 223]);
        imf_2 = imcrop(imf_, [231 1 223 223]);
        imf_3 = imcrop(imf_, [231 33 223 223]);
        imf_4 = imcrop(imf_, [0 33 224 224]);
        imf_c = imcrop(imf_, [116 17 223 223]);
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
        fresult = (repr1+repr2+repr3+repr4+repr5+repr6+repr7+repr8+repr9+repr10)/10;
        tempA = cat(4,tempA,fresult);
    end
    nReps = size(tempA,4);
    finA = mean(tempA,4);
%     disp(size(fresult));
%     disp(size(finA));
    res = vl_simplenn(net, finA) ;
    clbl = res(2).x;
    j = find(clbl==max(clbl));
    disp(j)
    testlabels = cat(1,testlabels,j);
end

% Saving Results
disp('Done classifying all scenes. Saving to CSV file');
B = idnames;
B = cat(2, B, testlabels);

filename = 'predTestLabels.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'ImgId,\t Prediction\n');
fclose(fid);
dlmwrite(filename, B, '-append', 'delimiter', '\t');

