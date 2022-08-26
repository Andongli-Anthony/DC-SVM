function [val,primals,duals,eduals] = iqph(H,c,A,b,E,d,params)

% [val,primals,duals,eduals] = iqph(H,c,A,b,E,d,params)
%
% A simple interior point algorithm for solving the quadratic program
%   maximize  - x'Hx/2 + c'x  st  Ax+b >= 0, Ex + d = 0
% when H is positive semidefinite.  Use H=[] for a linear program.
% E may be [], but A must be nonempty (otherwise it's a linear
% regression not a linear program).
%
% The dual to this program is
%   minimize  x'Hx/2 + b'y - d'z  st  y >= 0, A'y - Hx - E'z + c = 0
% We solve both programs at the same time, and return x, y, and z
% in primals, duals, and eduals respectively.  The objective value
% for the primal and dual programs is the same, and is returned in
% val.
%
% Detection of infeasibility/unboundedness is not very good: we just
% return NaN if it appears that the optimal solution to either the
% primal or the dual is very large.
% General robustness could also be improved.
%
% If last arg params is provided, it sets a few parameters (any
% fields not provided are defaulted):
%   params.verbose (bool): print messages?
%   params.maxiter (int): maximum number of iterations
%   params.canreorder (bool): is reordering for sparsity allowed?
%   params.epsilon (real): a small number
%   params.toobig (real): a large number
% For example, passing in struct('verbose',1) turns on messages
% without affecting any other defaults.

% Copyright 1998, 2005, 2010 Geoff Gordon

% do defaulting of parameters

if (nargin < 7)
  params = struct;
end

if (~isfield(params, 'epsilon'))
  params.epsilon = 1e-10;    % accuracy requirement for solution 
end

if (~isfield(params, 'toobig'))
  params.toobig = 1e+10;     % give up if solution gets too large 
end

if (~isfield(params, 'maxiter'))
  params.maxiter = 100;      % stop after this many iterations 
end

if (~isfield(params, 'canreorder'))
  params.canreorder = 1;    % set to zero to prevent reordering
end

if (~isfield(params, 'verbose'))
  params.verbose = 0;	    % prevent/allow messages
end

% get problem sizes
[m,n] = size(A);
if (size(b,1) ~= m || size(b,2) ~= 1)
    error('A: %d x %d, b: %d x %d\n', m, n, size(b,1), size(b,2));
end
if (size(c,1) ~= n || size(c,2) ~= 1)
    error('A: %d x %d, c: %d x %d\n', m, n, size(c,1), size(c,2));
end
[k,j] = size(E);
if (k == 0) 
  E = zeros(k,n);
  d = zeros(0,1);
end
if (m == 0)
  error('Must have at least one inequality constraint');
end

% default args
if (isempty(H)) H = (params.epsilon * .01) * speye(n); end
if (length(H)==1) H = H * speye(n); end
if (size(H,2)==1) H = spdiags(H,0,n,n); end

% symmetric part of H is all that matters
H = (H + H') / 2;

% There's no input argument controlling this, but the code allows for
% a quadratic penalty on the equality dual variables as well -- for
% now just defaulted to a very small number.  (Increasing it would
% yield "soft" equality constraints.)
He=[];
if (isempty(He)) 
  if (issparse(E) || (issparse(A) && issparse(H)))
    He = speye(k); 
  else
    He = eye(k); 
  end
  He = (params.epsilon * .01) * He;
end

% reorder for sparsity
if (params.canreorder & issparse(A) & issparse(H))
  order = symamd(H + A' * A);
  H = H(order,order);
  A = A(:,order);
  E = E(:,order);
  c = c(order);
else
  order = [];
end

% simple stupid initialization
primals = zeros(n,1);
duals = 100* ones(m,1);
slacks = 100* ones(m,1);
eduals = zeros(k,1);
complementarity = (slacks' * duals) / m;

% do until complementarity is small
iter = 0; 
feasresid = 1;
while ((complementarity + feasresid > params.epsilon) & (iter<params.maxiter))

  % compute and factor newton matrix
  dg = spdiags(duals ./ slacks, 0, m, m);
  nmat = (H + (A' * dg * A));
  nmat = [nmat E'; E -He];
  [nl, nd] = ldl(nmat);

  % compute affine scaling (predictor) step
  [dprimals, dduals, dslacks, deduals, steplen] = ...
  newtsys(nl, nd, 0, 0, c, A, b, E, d, H, He, primals, duals, slacks, eduals);

  % use Mehrotra's heuristic to pick a target point on the central path
  comp = ((slacks + steplen * dslacks)' * (duals + steplen * dduals)) / m;
  barrier = (comp/complementarity)^2 * comp;

  % compute corrected step
  ho = dslacks .* dduals;
  [dprimals, dduals, dslacks, deduals, steplen] = ...
    newtsys(nl, nd, barrier, ho, c, A, b, E, d, H, He, ...
    primals, duals, slacks, eduals);

  % do the update
  primals = primals + steplen * dprimals;
  slacks = slacks + steplen * dslacks;
  duals = duals + steplen * dduals;
  eduals = eduals + steplen * deduals;
  complementarity = (slacks' * duals) / m;
  feasresid = sum(abs(A * primals + b - slacks)) / m;
  if (k > 0)
    feasresid = feasresid + sum(abs(E * primals + d - He*eduals)) / k;
  end
  iter = iter + 1;

  % give up if numbers get too big
  if max(max(slacks),max(duals)) > params.toobig
    val = NaN;
    if (~isempty(order)) primals(order) = primals; end
    warning('iqph:illposed', 'program appears to be ill-posed');
    return;
  end

  % report progress
  if (params.verbose)
    fprintf('iter %d, step %g, target %g, actual %g, feas %g\n', ...
      iter, steplen, barrier, complementarity, feasresid);
  end

end

val = primals' * c - primals' * H * primals / 2;
if (~isempty(order)) primals(order) = primals; end

return


% [ -D -A  0  ]  dduals    = rhs
% [ -A' H  E' ]  dprimals  = rhs
% [  0  E  0  ]  deduals   = rhs
%
% where rhs is
%
%  A'*duals + c - H*primals
%  A*primals + b - slacks + duals - (mu + ho) ./ slacks
%  -(E*primals + d)

% Solve the Newton system of equations
%  -D  A  0  dduals    = rhs
%   A' H  E' dprimals  = rhs
%   0  E  0  deduals   = rhs
% (where D is diagonal and positive) which is the inner loop of
% our interior point method.
%
% We assume we are given already a factorization of the reduced
% Newton matrix
%   H+A'SA   E'
%     E      0
% where S=inv(D).  The reduced Newton matrix results from pivoting
% along D in the Newton system.

function [dprimals, dduals, dslacks, deduals, steplen] = ...
  newtsys(nl, nd, barrier, ho, c, A, b, E, d, H, He, ...
  primals, duals, slacks, eduals)

% a parameter
backoff = .99995;

m = length(duals);
n = length(primals);
k = length(eduals);

% compute primal and equality dual steps
rhs = c - H * primals + A' * (duals + (barrier - ho) ./ slacks - ...
  (duals ./ slacks) .* (A * primals + b));
sol = nl' \ ((nl \ [rhs-E'*eduals; -E*primals-d+He*eduals]) ./ nd);
dprimals = sol(1:n);
deduals = sol(n+1:n+k);
deduals = deduals(:);		%this prevents bombs when E==[]

% from primal step, compute dual and slack steps
dslacks = A * (primals + dprimals) + b - slacks;
dduals = (barrier - ho - duals .* dslacks) ./ slacks - duals;

% this is more accurate but much slower
%dduals = [spdiags(slacks,0,m,m); A'] \ ...
%	 [(barrier - ho - duals .* dslacks) - duals .* slacks; ...
%	  - c - A'*duals + E'*(eduals+deduals)];

% compute stepsize: longest possible w/o getting too close to zero
steplen = max(-dduals ./ duals);
steplen = max(steplen, max(-dslacks ./ slacks));
if (steplen <= 0) steplen = Inf; else steplen = 1 / steplen; end
steplen = min(1, .666 * steplen + (backoff - .666) * steplen^2);

return


% [l,d] = ldl(a)
%
% Decompose the matrix A as LDL', where L is lower triangular and D is
% diagonal (returned as a vector).  A must be symmetric quasidefinite.

function [l,d] = ldl(a)

n = min(size(a));

d = zeros(n,1);
l = zeros(n,n);

for i = 1:n
  if (i > 1)
    x = l(i:n,1:(i-1)) * (d(1:(i-1)) .* l(i,1:(i-1))');
  else
    x = 0;
  end
  x = a(i:n,i) - x;
  d(i) = x(1);
  l(i:n,i) = x / d(i);
end

return


% Usage example: build and solve a Markov decision process.  The MDP
% has n states, 2 actions.  One action shifts state down by 1, the
% other shifts it up by 1.  Discretized Gaussian noise with 3*sigma=k
% states is added to each move.  Goal states are 1 and n, and edge
% costs are 1+noise.  Initial state distribution is uniform.  The true
% value function (primal variables, plotted at end) should be
% approximately min(i,n+1-i) for state i; the true edge visitation
% frequencies (dual variables, also plotted at end) are approximately
% linear for edges on the way from the center to the goals (except
% very near the goals, where we have a chance of skipping some states
% due to noise), and 0 for other edges.
n = 1000;
k = 6;
gauss = -3:6/(2*k):3;
gauss = exp(-gauss.^2/2);
gauss = gauss / sum(gauss);
right = spdiags(repmat(gauss, [n 1]), -k+1:k+1, n, n);
left = spdiags(repmat(gauss, [n 1]), -k-1:k-1, n, n);
edge = [right-speye(n); left-speye(n)];
cost = 1 + .1*randn(2*n,1);
init = ones(n,1)/n;
[v,p,d] = iqph([], init, edge, cost, [], [], struct('canreorder',0,'verbose',1));

figure(1);
plot(1:n,p)
figure(2);
plot(1:n,d(1:n),1:n,d(n+1:end))
