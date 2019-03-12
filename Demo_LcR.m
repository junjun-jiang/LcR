
% =========================================================================
% Face Hallucination via Locality-constrained Representation
% Example code
%
% Reference
% [1] Junjun Jiang, Ruimin Hu, Zhen Han, Tao Lu, Kebin Huang, ¡°Position-Patch Based 
% Face Hallucination via Locality-constrained Representation,¡± in the International 
% Conference on Multimedia and Expo (ICME 2012), Melbourne, Australia, pp. 212-217, Jul 2012.
%
% [2]Junjun Jiang, Ruimin Hu, Zhongyuan Wang, and Zhen Han, "Noise Robust Face 
% Hallucination via Locality-constrained Representation," TMM, 2014
%
% For any questions, email me by jiangjunjun@whu.edu.cn
%=========================================================================



clc;close all;
clear all;
addpath('.\utilities');

% set parameters
nrow        = 120;        % rows of HR face image
ncol        = 100;        % cols of LR face image
nTraining   = 360;        % number of training sample
nTesting    = 40;         % number of ptest sample
upscale     = 4;          % upscaling factor 
BlurWindow  = 4;          % size of an averaging filter 
tau         = 0.04;       % locality regularization
patch_size  = 12;         % image patch size
overlap     = 4;          % the overlap between neighborhood patches

YH          = zeros(nrow,ncol,nTraining); 
YL          = zeros(nrow,ncol,nTraining); 

bb_psnr    = zeros(1,nTesting);
sr_psnr    = zeros(1,nTesting);
bb_ssim    = zeros(1,nTesting);
sr_ssim    = zeros(1,nTesting);


% generate the training and testing samples from the FEI Face Database
% FEIDBResize(nrow,ncol,nTraining);% we have genetaed the  training and testing samples from 
% FEI Face Database, and you can download it from 'http://fei.edu.br/~cet/facedatabase.html' also.

% construct the HR and LR training pairs from the FEI face database
[YH YL] = Training_LH(upscale,BlurWindow,nTraining);

fprintf('\nface hallucinating for %d input test images\n', nTesting);

for TestImgIndex = 1:nTesting

    fprintf('\nProcessing  %d/%d LR image\n', TestImgIndex,nTesting);

    % read ground truth of one test face image
    strh    = strcat('.\testFaces\',num2str(TestImgIndex),'_test.jpg');
    im_h    = imread(strh);

    % generate the input LR face image by smooth and down-sampleing
    w       = fspecial('average',[BlurWindow BlurWindow]);
    im_s    = imfilter(im_h,w);
    im_l    = imresize(im_s,1/upscale,'bicubic');
    im_l    = double(im_l);
%     figure,imshow(im_l);title('input LR face');

    % face hallucination via LcR
    [im_SR] = LcRSR(im_l,YH,YL,upscale,patch_size,overlap,tau);
    [im_SR] = uint8(im_SR);


    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');

    % compute PSNR and SSIM for Bicubic and our method
    bb_psnr(TestImgIndex) = psnr(im_b,im_h);
    bb_ssim(TestImgIndex) = ssim(im_b,im_h);

    sr_psnr(TestImgIndex) = psnr(im_SR,im_h);
    sr_ssim(TestImgIndex) = ssim(im_SR,im_h);

    % display the objective results (PSNR and SSIM)
    fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr(TestImgIndex));
    fprintf('PSNR for LcR Recovery: %f dB\n', sr_psnr(TestImgIndex));
    fprintf('SSIM for Bicubic Interpolation: %f dB\n', bb_ssim(TestImgIndex));
    fprintf('SSIM for LcR Recovery: %f dB\n', sr_ssim(TestImgIndex));

    % show the images
%     figure, imshow(im_b);
%     title('Bicubic Interpolation');
%     figure, imshow(uint8(im_SR));
%     title('LcR Recovery');
    
    % save the result
    strw = strcat('./results/',num2str(TestImgIndex),'_SR.bmp');
    imwrite(uint8(im_SR),strw,'bmp');
end


fprintf('===============================================\n');
fprintf('Average PSNR of Bicubic Interpolation: %f\n', sum(bb_psnr)/nTesting);
fprintf('Average PSNR of LcR method: %f\n', sum(sr_psnr)/nTesting);
fprintf('Average SSIM of Bicubic Interpolation: %f\n', sum(bb_ssim)/nTesting);
fprintf('Average SSIM of LcR method: %f\n', sum(sr_ssim)/nTesting);
fprintf('===============================================\n');


