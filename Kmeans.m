%---k means
f=load('/MATLAB/work/AllSamples.mat');
samples = cell2mat(struct2cell(f));
centroids_st1 ={};
random_k=[];
d=[];
sel_c=[];
result=[];
c=[];
%plot(samples(:,1),samples(:,2),'+b'); %plot datapoints
hold on;
%set 1 
random_c(1,:) = samples(7,:);
random_c(2,:) = samples(119,:);
random_c(3,:) = samples(90,:);
random_c(4,:) = samples(22,:);
random_c(5,:) = samples(36,:);
random_c(6,:) = samples(69,:);
random_c(7,:) = samples(108,:);
random_c(8,:) = samples(220,:);
random_c(9,:) = samples(21,:);
random_c(9,:) = samples(250,:);
random_c(10,:) = samples(47,:);
fprintf('OBJECTIVE FUNCTION VALUES\n');
fprintf('Strategy 1 Set#1\n');
result = strategy_1(random_c,samples);
fprintf('%.0f\n',result);
figure('Name','Strategy 1','NumberTitle','off')
plot(2:10,result,'r');
hold on
%set 2 (uncomment to test for set 2)
random_c(1,:) = samples(7,:);
random_c(2,:) = samples(118,:);
random_c(3,:) = samples(90,:);
random_c(4,:) = samples(22,:);
random_c(5,:) = samples(16,:);
random_c(6,:) = samples(70,:);
random_c(7,:) = samples(108,:);
random_c(8,:) = samples(220,:);
random_c(9,:) = samples(20,:);
random_c(9,:) = samples(250,:);
random_c(10,:) = samples(47,:);
fprintf(' Set#2\n');
result=strategy_1(random_c,samples);
fprintf('%.0f\n',result);
xlabel('number of clusters k');
ylabel('objective function value');
plot(2:10,result,'b');
grid

function [result] =  obj(samples,random_k)
%-----------------------------------------
%function to find the objective function values
%input: data samples,final centroid values which are found out by k means
%output: objective function vector for k=2....10
%-----------------------------------------
  result=0;
  cluster=[[]];
  A=[[]];
  for i = 1:size(samples,1)
    for j = 1:size(random_k,1)
        A(i,j)= ((samples(i,1)-random_k(j,1))^2) + ((samples(i,2)-random_k(j,2))^2);    
    end
  end
  [val,loc] = min(A');
  for i = 1:size(samples,1)
      cluster=[cluster;samples(i,:),loc(i)'];
      result = result + val(i)';
  end
 
end

function [loc] = sel(samples,sel_c)
%-----------------------------------------
%input: data samples,current centroid values
%output: location of next centroid value
%-----------------------------------------
D=[[]];
  for i = 1:size(samples,1)
      for j=1:size(sel_c,1)
           D(i,j)= sqrt((samples(i,1)-sel_c(j,1))^2) + ((samples(i,2)-sel_c(j,2 ))^2); 
      end    
  end
  
  if size(D,2)>1
      m=mean(D');
  else
      m=D';
  end
  [val,loc] = max(m);%finding the point with the maximum average distance form the given points
  samples(loc,:);  
end
function [c]= k_means(random_k,samples)
%-----------------------------------------
%function to implement k means algorithm
%input: centroid values, data samples
%output: final centroid values
%-----------------------------------------
D=[[]];

for i = 1:size(samples,1)   
        for j=1:size(random_k,1)
           D(i,j)= sqrt((samples(i,1)-random_k(j,1))^2) + ((samples(i,2)-random_k(j,2 ))^2); 
        end    
end
  
[val,loc] = min(D');
D = [loc' samples];
c=[];
flag=0;
for j = 1:size(random_k,1)
    ind = D(:,1) == j;
    A = D(ind,:);%grouping data points
    c=[c;mean(A(:,2)) mean(A(:,3))];%finding centroid for the group of data points
end           
c(isnan(c))=0;
%c=c(all(~isnan(c), 2),:);
if isequal(c,random_k)==0 
    [c]= k_means(c,samples);
else
   return; 
end
end

function [result]= strategy_1(random_c,samples)
%-----------------------------------------
%function to implement k means clustering and to find objective function values
%input: initial random centroid values, data samples
%output: objective function vector for k=2....10
%-----------------------------------------
centroids_st1={};
result=[];
c_k=[];
c=[];
for i = 2:10
    random_k = [random_c(1:i,:)];
    [c]= k_means(random_k,samples) ;
    centroids_st1{i}=[c];
end
colr=['o','r','k','g','y','c'];
%to plot the corresponding centroids 
% for i = 2:6
%     c=centroids_st1{i};
%     hold on
%     plot(c(:,1),c(:,2),'ob','color',colr(i));
% end
% for i = 7:10
%     c=centroids_st1{i};
%     hold on
%     plot(c(:,1),c(:,2),'+b','color',colr(i-5));
% end  
for k=2:10
    c_k = [c(1:k,:)];
    result = [result obj(samples,c_k)];    
end
end

end

