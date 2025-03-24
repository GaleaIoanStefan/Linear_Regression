clc; clear; close all;

% Loading the input data and storing it into easier-to-work-with variables
load("proj_fit_14.mat");
x1_id = id.X{1,1};
x2_id = id.X{2,1};
x1_val = val.X{1,1};
x2_val = val.X{2,1};
y_id = id.Y;
y_val = val.Y;
dim_id = id.dims(1,1);
dim_val = val.dims(1,1);

% Choosing maximum polynomial degree
nmax = 26;

% Initialising the phi matrices with a column vector of ones
phi_id = ones(dim_id*dim_id,1);
phi_val = ones(dim_val*dim_val,1);

% Creating polynomial approximations of different degrees using linear regression
for n = 1:nmax
    % Identification
    phi_id = phi_calc(phi_id, x1_id, x2_id, n);
    yflat_id = y_flattening(dim_id, y_id);
    theta = phi_id\yflat_id;
    yhat_id = phi_id * theta;
    
    % Validation
    phi_val = phi_calc(phi_val, x1_val, x2_val, n);
    yflat_val = y_flattening(dim_val, y_val);
    yhat_val = phi_val * theta;

    % Turning yhat_id into a matrix
    for i = 1:dim_id
       for j = 1:dim_id
            ymatrix_id(i,j) = yhat_id((i-1)*dim_id+j,1);
        end
    end

    % Turning yhat_val into a matrix
    for i = 1:dim_val
       for j = 1:dim_val
            ymatrix_val(i,j) = yhat_val((i-1)*dim_val+j,1);
        end
    end

    % Storing all the approximated output matrices into larger matrices so
    % that only the best approximator to be plotted afterwards
    yapprox_id(((n-1)*dim_id+1):(n*dim_id),:) = ymatrix_id;
    yapprox_val(((n-1)*dim_val+1):(n*dim_val),:) = ymatrix_val;


    % Caculating the Mean Square Error for the Identification Set
    for k = 1:dim_id
        err_id(k) = (yflat_id(k) - yhat_id(k))^2;
    end
    MSE_id(n) = 1/dim_id * sum(err_id);


    % Caculating the Mean Square Error for the Validation Set
    for k = 1:dim_val
        err_val(k) = (yflat_val(k) - yhat_val(k))^2;
    end
    MSE_val(n) = 1/dim_val * sum(err_val);

end

% Finding the minimum MSE for the Identification Set
[minimum_id, min_index_id] = min(MSE_id);
disp("The minimum Mean Square Error for the Identification Set is MSE_id =")
disp(minimum_id)
disp("And it is achieved for a polynomial approximation of degree:")
disp(min_index_id)


% Finding the minimum MSE for the Validation Set
[minimum_val, min_index_val] = min(MSE_val);
disp("The minimum Mean Square Error for the Validation Set is MSE_val =")
disp(minimum_val)
disp("And it is achieved for a polynomial approximation of degree:")
disp(min_index_val)

% Plotting the MSE
% Identification MSE plot depending on polynomial degree
figure
q = 1:length(MSE_id);
plot(q,MSE_id); grid
hold on
plot(min_index_id,minimum_id,'*')
legend('MSE value', 'Minimum MSE')
xlabel('Polynomial Degree')
ylabel('MSE')
title("Identification Set MSE");
hold off

% Validation MSE plot depending on polynomial degree
figure
q = 1:length(MSE_val);
plot(q,MSE_val); grid
hold on
plot(min_index_val,minimum_val,'*')
legend('MSE value', 'Minimum MSE')
xlabel('Polynomial Degree')
ylabel('MSE')
title("Validation Set MSE");
hold off

% Plotting the best polynomial approximations with respect to the input data
% For the Identification Set
figure
hold on
mesh(x1_id,x2_id,y_id,'EdgeAlpha', 1, 'FaceAlpha',0.4); colormap turbo;
mesh(x1_id,x2_id,yapprox_id(((min_index_id-1)*dim_id+1):(min_index_id*dim_id),:),'FaceColor','interp');
legend('Initial Data', 'Approximated polynomial representation')
title("Identification Set");
grid
hold off

% For the Validation Set
figure
hold on
mesh(x1_val,x2_val,y_val, 'EdgeAlpha', 1, 'FaceAlpha',0.4); colormap hot;
mesh(x1_val,x2_val,yapprox_val(((min_index_val-1)*dim_val+1):(min_index_val*dim_val),:),'FaceColor','interp');
legend('Initial Data', 'Approximated polynomial representation')
title("Validation Set");
grid
hold off

% Function that returns the flattened y matrix and the computed value of
% the phi matrix
function phi = phi_calc(phi, x1, x2, n)
    dim = length(x1);
    for i = 1:dim
        for j = 1:dim
            for k = 1:n+1
               phi_concat((i-1)*dim + j,k) = x1(i)^(n-k+1)*x2(j)^(k-1);
            end
        end
    end
    phi = [phi, phi_concat];
end

% Function that flattens the y matrix, converting it from a square matrix
% to a column vector for the liear regression algorithm
function y_flat = y_flattening(dim, y)    
    for i = 1:dim
        for j = 1:dim
            y_flat((i-1)*dim+j,1) = y(i,j);
        end
    end
end