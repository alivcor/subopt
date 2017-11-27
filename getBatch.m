function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
disp(size(im))
%im = 256 * reshape(im, 32, 32, 1, []) ;
labels = imdb.images.label(:,batch) ;