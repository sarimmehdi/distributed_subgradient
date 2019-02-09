function [x_new,x_hat,v_hat,conv_error,outmemory_num,outmemory_den,dual_cost] = node(C,B,a,A,b,Aineq,bineq,Aeq,beq,v,xInput,k,inmemory_num,inmemory_den,min,max,I)
%C = f(x) = C'*x
%B = factor between 1 and 0 (preferably really small) that influences your
%gradient step-size c
%a = the row of the adjacency matrix corresponding to your node
%A, b = vectors of your inequality constraints
%v = matrix of all the dual variable vectors
%k = the current iteration of your distributed algorithm
%memory_num and memory_den are useful for calculating the estimate of the
%state
 
%compute weighted average vector of your dual variables vector
l_t = bsxfun(@times,v,a); l = sum(l_t,2);

%compute dual cost
dual_cost = C*xInput + l'*A*xInput - l'*(b/I);

%compute next iteration of x
f = C + l'*A;
options = optimoptions('linprog','Display','none');
x_new = linprog(f,Aineq,bineq,Aeq,beq,min,max,options);

%find the gradient step-size
c = B / (k + 1);
 
%find dual variable at next time instant (must be projected onto set of all
%positive real numbers, so make all non-positive numbers 0)
v_hat = l + c*A*x_new - c*(b/I);
v_hat(v_hat<=0) = 0; conv_error = vecnorm(v_hat - l);

%compute an estimate of your state at time k+1
outmemory_num = inmemory_num + c*x_new;
outmemory_den = inmemory_den + c; 
x_hat = outmemory_num/outmemory_den;
