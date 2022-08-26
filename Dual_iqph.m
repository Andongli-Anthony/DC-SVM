% Run SVM on given data x and y, and return w, b, and index of support
% vectors
% K: m by m kernel matrix, dot products of m examples
% y: m by 1 vector, containing labels (+1/-1) for the m examples
% C: a scalar, penalty for violation of the margin
% w: n by 1 vector, each coefficient corresponds to one feature
% b: a scalar, the intercept for decision function: wx_i + b
% support: indices of the support vectors in the m examples
function [alpha,support,time] = Dual_iqph(K, y, C)
    if size(K,1)~=size(K,2)
        disp('ERROR, kernel matrices must be m by m (m: #examples).');
        return;
    end
    if (size(K,1) ~= length(y))
        disp('ERROR: K and y must be for same number of examples.');
    end
    
    m = size(K,1);
    y = y(:); % make sure y is m by 1, not 1 by m

    H = K ;  
    f = -zeros(m,1); 
    A = [eye(m); -eye(m)];
    b = [zeros(m,1); ones(m,1)*C];
    Aeq = ones(1,m);   
    beq =- 1;   


    disp('Start timer in the dual form QP solver ...');
    tic
    [val,alpha,duals,eduals] = iqph(H,f,A,b,Aeq,beq);
    time = toc;
    fprintf(1, 'Finish solving the dual form QP, time usage: %d.\n', time);   
    eps = .00001;
    support = find(alpha > eps);
end
