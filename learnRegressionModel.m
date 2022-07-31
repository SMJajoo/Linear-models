function model_parameters=learnRegressionModel(predictors, temp)

%%  first order polynomial is a linear function: for three variables: y= a1X1+a2X2+a3X3+b
%% for brute force approach, we need to check every combinations of a1, a2, a3 and using for loops
%% Which almost impossible to compute. You can try. 




%% Optimisation using gradient descent Check lecture slides in "Gradient Descent" page 6
%initialise the parameters a and b.
a=zeros(size(predictors,2),1);  % slope of linear equations, the number should be same as the number of features
b=0;      % a constant for the linear equation.
lR=0.05; % learning rate
thr=0.01; % stopping threshold
for iter=1:1000 % run for maximum of 1000 iterations
    % calculate the derivatives for each parameter
    det_a=zeros(size(predictors,2),1);
    for i=1: size(predictors, 2)
        predict_y=zeros(size(temp,1),1);
        for j=1:length(a)  % calculate predicted y using the current parameters
            predict_y=predict_y+a(j)*predictors(:,j);
        end
        predict_y=predict_y+b;
        det_a(i)= mean(predictors(:,i).*(predict_y-temp));  % partial derivative for a(i), mean of x(y_pred-y)
    end
    det_b   = mean(predict_y-temp);   % partial derivative for b
    
    % update parameters a and b
    a=a-lR*det_a;
    b=b-lR*det_b;
    
    y_prediction=[min(predictors(:,1)):max(predictors(:,1))]*a(1)+b;
    figure(1);plot(predictors(:,1),temp,'or');
    hold on;plot([min(predictors(:,1)):max(predictors(:,1))],y_prediction,'b');
    title("iteration=", iter);
    pause(0.01);
    
    if max(abs(det_a))<thr && abs(det_b)<thr  % if the updated parameter is small enough then stop the optimisation
        break;
    end
    
end

model_parameters=[a;b];




