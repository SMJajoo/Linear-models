function label=knn(training_instances, training_labels, test_instance, k)

label=zeros(size(test_instance,1),1);
for i=1: size(test_instance,1)
    difF=training_instances- repmat(test_instance(i,:),size(training_instances,1),1);
    EuclideanDifF=sqrt(sum(difF.*difF,2));  %Euclidean distance of the feature. 
    [sortValue,sortIndx]=sort(EuclideanDifF);
    nearestK=sortIndx(1:k);   % Get indeces of top K nearest instrances
    nKLabels=training_labels(nearestK); % Retrieve their labels.
    label(i)=mode(nKLabels); % Take the most frequent value in nKLabels as the predicted label.
end