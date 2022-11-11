
f=load('/MATLAB/work/mnist/train_0_img.mat');
train_0_img = cell2mat(struct2cell(f)); %training images for digit '0'
[avg_br_train_0, avg_var_train_0]=feature_extract(train_0_img);
%a=size(avg_br_train_0)

f=load('/MATLAB/work/mnist/train_1_img.mat');
train_1_img = cell2mat(struct2cell(f));%training images for digit '1'
[avg_br_train_1, avg_var_train_1]=feature_extract(train_1_img);
%b=size(avg_br_train_1)

f=load('/MATLAB/work/mnist/test_0_img.mat');
test_0_img = cell2mat(struct2cell(f)); %test images for digit '0'
[avg_br_test_0,avg_var_test_0]=feature_extract(test_0_img);
%c=size(avg_br_test_0)

f=load('/MATLAB/work/mnist/test_1_img.mat');
test_1_img = cell2mat(struct2cell(f));%test images for digit '1'
[avg_br_test_1,avg_var_test_1]= feature_extract(test_1_img);

avg_br_test = [avg_br_test_0;avg_br_test_1];
avg_var_test= [avg_var_test_0;avg_var_test_1];

f=load('/MATLAB/work/mnist/test_0_label.mat');
Test_0_label = cell2mat(struct2cell(f));
f=load('/MATLAB/work/mnist/test_1_label.mat');
Test_1_label = cell2mat(struct2cell(f));
Test_label = [Test_0_label;Test_1_label];

%normal distribution parameters for training set features
disp('mean of feature 1 of image training set for digit 0');
mu_br_train_0=mean(avg_br_train_0) %mean of feature 1 of image training set for digit 0
disp('mean of feature 1 of image training set for digit 1');
mu_br_train_1=mean(avg_br_train_1) %mean of feature 1 of image training set for digit 1
disp('mean of feature 2 of image training set for digit 0');
mu_var_train_0=mean(avg_var_train_0) %mean of feature 2 of image training set for digit 0
disp('mean of feature 2 of image training set for digit 1');
mu_var_train_1=mean(avg_var_train_1) %mean of feature 2 of image training set for digit 1
disp('standard deviation of feature 2 of image training set for digit 0');
sigma_br_train_0=std(avg_br_train_0) %variance of feature 1 of image training set for digit 0
disp('standard deviation of feature 2 of image training set for digit 1')
sigma_br_train_1=std(avg_br_train_1) %variance of feature 1 of image training set for digit 1
disp('standard deviation of feature 2 of image training set for digit 0')
sigma_var_train_0=std(avg_var_train_0) %variance of feature 2 of image training set for digit 0
disp('standard deviation of feature 2 of image training set for digit 1')
sigma_var_train_1=std(avg_var_train_1) %variance of feature 2 of image training set for digit 1

pre_Result=[]; %predicted test label
for i = 1:size(Test_label,1)
    pre_Result(i)=nb_classify(avg_br_test(i,1), avg_var_test(i,1),mu_br_train_0,sigma_br_train_0,mu_var_train_0,sigma_var_train_0,mu_br_train_1,sigma_br_train_1,mu_var_train_1,sigma_var_train_1);
end
size(pre_Result)
disp('Classification accuracy for Overall test set in percent');
[accuracy] = comput(Test_label',pre_Result) %accuracy parameters for overall test data
disp('Classification accuracy for digit ''0'' test set');
[accuracy_0] = comput(Test_0_label',pre_Result(1:size(Test_0_label,1)))%accuracy parameters for "0"
disp('Classification accuracy for digit ''1'' test set');
[accuracy_1] = comput(Test_1_label',pre_Result(size(Test_0_label,1)+1:end)) %accuracy parameters for "1"

function [f1,f2]= feature_extract(img_mat)
%-----------------------------------------------%
%function to extract features from image data set
%input: image matrix
%output: features f1 and f2 for each image in image matrix
%-----------------------------------------------%
     f1=[];
     f2=[];
     s= size(img_mat);
     img_mat_2d= reshape(img_mat,s(1)*s(2),s(3));%converting 3D image matrix to 2D
     img_mat_2d=img_mat_2d'; 
     %f1: first feature, average brightness of the image
     f1 = mean(img_mat_2d,2);
     %f2: second feature, The average of the variances of each rows of the image.
     for i =1:s(3)
        c = img_mat(:,:,i)'; 
        f2(i,1)=mean(var(c));
     end
end

function p = nb_classify(f1,f2,mu_f1_0,sigma_f1_0,mu_f2_0,sigma_f2_0,mu_f1_1,sigma_f1_1,mu_f2_1,sigma_f2_1)
%-----------------------------------------------------------------------------%
%function for naive bayes classifier
%input:feature set for test data,normal distribution parametes: mu and variance
%for training set (digit 0 and digit 1)
%output:predicted value of test label after naive bayes classification
%-----------------------------------------------------------------------------%
    P_y =0.5; %prior (given)
    
    %calculating likelihood of test digit being 0
    f= (f1 - mu_f1_0);
    p_f1_y = (1/sqrt((2*pi*(sigma_f1_0)^2)))*exp((-1)*(f^2)/(2*(sigma_f1_0)^2));
    f= (f2 - mu_f2_0);
    p_f2_y = (1/sqrt((2*pi*(sigma_f2_0)^2)))*exp((-1)*(f^2)/(2*(sigma_f2_0)^2));
    P_y0_x= p_f1_y*p_f2_y*P_y;
    
    %calculating likelihood of test digit being 1
    f= (f1 - mu_f1_1);
    p_f1_y = (1/sqrt((2*pi*(sigma_f1_1)^2)))*exp((-1)*(f^2)/(2*(sigma_f1_1)^2));
    f= (f2 - mu_f2_1);
    p_f2_y = (1/sqrt((2*pi*(sigma_f2_1)^2)))*exp((-1)*(f^2)/(2*(sigma_f2_1)^2));
    P_y1_x= p_f1_y*p_f2_y*P_y;
    
    %compare the probabilities to determine the class of test digit
    if (P_y0_x > P_y1_x) 
        p= 0;
    else
        p= 1;
    end
end

function [a] = comput(act_Result,pre_Result)
%-----------------------------------------------------------------------%
%function for calculating the accuracy matrices
%input: actual test labels, prediced test labels
%output: values for accuracy of the classifier
%------------------------------------------------------------------------%
    tp = 0; %true positive
    tn = 0; %true negative
    fp = 0; %false positive
    fn = 0; %false negative
    for j= 1:size(pre_Result,2)
         if act_Result(1,j) == 1 && pre_Result(1,j) == 1
             tp = tp+1;
         elseif act_Result(1,j) == 0 && pre_Result(1,j) == 0
             tn = tn+1;  
         elseif act_Result(1,j) == 0 && pre_Result(1,j) == 1
             fp = fp+1;
         elseif act_Result(1,j) == 1 && pre_Result(1,j) == 0
             fn = fn+1;
         end
    end
    a= ((tp+tn)/(tp+tn+fp+fn))*100; %accuracy
    %p = tp/(tp+fp);% precision
    %r= tp/(tp+fn);%recall
    %f=(2*p*r)/(p+r);%f_score
end
    
    
