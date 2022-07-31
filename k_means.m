function cluster_labels=k_means(data_points, k)

colorM={'or','og','ob','oc','ok','om','oy'};  % some colour coding for plotting, only support 7 class for now.
%initialise the mean point of each cluster, the random mean points need to
%be between min and max feature values. 
minX=min(data_points);
maxX=max(data_points);
randMeanX=repmat(minX,k,1)+rand(k,size(data_points,2)).*repmat((maxX-minX), k,1); % randMeanX is k by F, F is feature dimension

% Iteratively update the centre of cluster and assign data points until
% convergence
cluster_labels=zeros(size(data_points,1),1); 
New_cluster_labels=cluster_labels;
for i=1:50   % set a maximum of 50 iterations
    
    for indxP=1: size(data_points,1) % for each of the data_points    
        minD=inf;
        for indxC=1:k                % for each cluster
            meanX=randMeanX(indxC,:);   %obtain the current cluster centre
            difX=(data_points(indxP,:)-meanX);
            EdX=sqrt(sum(difX.*difX));    % Euclidean distance to the centre
            if EdX<minD
                New_cluster_labels(indxP)=indxC;
                minD=EdX;
            end
        end
    end
    % Update the mean of each cluster and visualise the current clustering
    if sum(abs(New_cluster_labels-cluster_labels))==0
        break;                                        % stop if no more label updating
    else
        cluster_labels=New_cluster_labels;
        clf(figure(1));figure(1);
        for indxC=1:k
            indxP=find(cluster_labels==indxC); % find index of point belongs to each cluster.
            randMeanX(indxC,:)=mean(data_points(indxP,:)); % calculate the new cluster centre.
            plot(data_points(indxP,1),data_points(indxP,2),colorM{indxC});hold on;
        end
        title("iteration=",i);
        pause();
    end
    
end

