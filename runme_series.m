%0 means you want to use the result from linprog for comparison
noLinprog = 0;

%generate random matrices (make sure to make noLinprog 1 for values of I greater than 500)
%set second argument to 0 if you don't want to use original data as Vujanic
%if third argument is 1, then you are going to have more cars with higher
%power (recommended that you keep it as 1)
[I,P,Emin,Emax,Einit,Eref,rateu,ratev,deltaT,N,Pmax,Pmin,Cu,Cv,deltau,deltav] = MakeRandom(100,4,1);
rng('shuffle');

if noLinprog == 0
    %first find the cost and violation using linprog. This allows us to observe
    %how close the dual subgradient's result goes to linprog's result as number
    %of iterations increase (keep in mind that for too high value of I, the size
    %of matrices becomes too large and you have memory issue, thus my above
    %recommendation. This is also a good motivation for using a distributed algorithm
    %in the first place)
    b = [Pmax*ones(1,N) -Pmin*ones(1,N)];
    c = []; A = []; A_eq = []; b_eq = []; A_ineq = []; b_ineq = []; X_tu = []; X_tl = [];
    tempMat1 = [eye(N,N); zeros(1,N)]; tempMat2 = zeros(N+1,N); tempMat2(N+1,N) = -1;
    %setup matrices for use in linprog
    tic
    fprintf('Now creating matrices\n')
    for ii=1:I
        c = [c P(1,ii)*(Cu(ii,:)+deltau(1,ii)) -P(1,ii)*(Cv(ii,:)+deltav(1,ii)) zeros(1,N)];
        mata = P(1,ii)*eye(N,N); matb = -P(1,ii)*eye(N,N); tempA = [mata matb zeros(N,N); matb mata zeros(N,N)];
        A = [A tempA];
        matA = tril(-P(1,ii)*deltaT*rateu(1,ii)*ones(N,N)); matB = tril(P(1,ii)*deltaT*ratev(1,ii)*ones(N,N));
        tempa = [matA matB eye(N,N)];
        if ii > 1 && ii < I
            A_eq = [A_eq; zeros(N,3*N*(ii-1)) tempa zeros(N,3*N*(I-ii))]; 
            A_ineq = [A_ineq; zeros(N+1,3*N*(ii-1)) tempMat1 tempMat1 tempMat2 zeros(N+1,3*N*(I-ii))];
        elseif ii == I
            A_eq = [A_eq; zeros(N,3*N*(ii-1)) tempa]; 
            A_ineq = [A_ineq; zeros(N+1,3*N*(ii-1)) tempMat1 tempMat1 tempMat2];
        else
            A_eq = [A_eq tempa zeros(N,3*N*(I-ii))]; 
            A_ineq = [A_ineq tempMat1 tempMat1 tempMat2 zeros(N+1,3*N*(I-ii))];
        end
        b_eq = [b_eq Einit(1,ii)*ones(1,N)]; b_ineq = [b_ineq ones(1,N) -Eref(1,ii)];
        X_tl = [X_tl zeros(1,2*N) Emin*ones(1,N)]; X_tu = [X_tu ones(1,2*N) Emax(1,ii)*ones(1,N)];
        if rem(ii,10)==0
            fprintf('Matrix creation progress (in percentage): %f\n',(ii/I)*100)
        end
    end
    linprogMatrixCreationTime = toc;   %call this variable in command window to see actual time
    if linprogMatrixCreationTime <= 60
        fprintf('Time taken to create matrices: %f seconds\n',toc)
    else
        fprintf('Time taken to create matrices: %f minutes\n',toc/60)
    end
    linprogMatrixCreationTime = toc/60;   %call this variable in command window to see actual time
    
    %centralized means I combine global and local inequalities into one
    A_ineq = [A; A_ineq]; b_ineq = [b b_ineq];
    tic
    x_final = linprog(c,A_ineq,b_ineq,A_eq,b_eq,X_tl,X_tu);
    linprogSolveTime = toc; %call this variable in command window to see actual time
    if linprogSolveTime <= 60
        fprintf('Time taken to solve using linprog: %f seconds\n',toc)
    else
        fprintf('Time taken to solve using linprog: %f minutes\n',toc/60)
    end
    linprogSolveTime = toc/60; %call this variable in command window to see actual time
    cost = c*x_final
    violation = A_ineq*x_final - b_ineq; sumC = zeros(24,1);
    for t=1:24
        sumC(t,1) = sumC(t,1) + violation(t,1) - Pmax;
    end
    average_violation = sumC/24
    %convert your linear solution into a matrix for comparison with distributed
    %solution
    x_star = [];
    for ii=1:I
        x_star = [x_star x_final(3*N*ii-3*N+1:3*N*ii,:)];
    end
end

%distributed algorithm starts from here
%k represents the total number of steps you take before algorithm stops
%you can only change k and beta!
fprintf('Creating matrices for distributed algorithm\n')
tic
k=5000; beta=0.01; convergence_error = 0.001;
mem_den=zeros(1,I); newmem_den = mem_den; v=zeros(48,I);
mem_num=zeros(3*N,I); newmem_num = mem_num;
x = zeros(3*N,I); xEst = x; xBet = x;

%create A, b and c matrix (cost in c matrix is perturbed by randomly generated numbers deltau, deltav)
b = [Pmax*ones(N,1); -Pmin*ones(N,1)]; dual_history = cell(I,1);
A = cell(I,1); A_ineq = cell(I,1); b_ineq = cell(I,1); A_eq = cell(I,1); b_eq = cell(I,1); X_tu = cell(I,1);
c = zeros(I,3*N); tempMat1 = [eye(N,N); zeros(1,N)]; tempMat2 = zeros(N+1,N); tempMat2(N+1,N) = -1;
for ii=1:I
    mata = P(1,ii)*eye(N,N); matb = -P(1,ii)*eye(N,N);
    A{ii} = [mata matb zeros(N,N); matb mata zeros(N,N)];
    matA = tril(-P(1,ii)*deltaT*rateu(1,ii)*ones(N,N)); 
    matB = tril(P(1,ii)*deltaT*ratev(1,ii)*ones(N,N));
    A_eq{ii} = [matA matB eye(N,N)]; b_eq{ii} = Einit(1,ii)*ones(N,1);
    A_ineq{ii} = [tempMat1 tempMat1 tempMat2]; b_ineq{ii} = [ones(N,1); -Eref(1,ii)];
    X_tu{ii} = [ones(1,2*N) Emax(1,ii)*ones(1,N)];
    c(ii,:) = [P(1,ii)*(Cu(ii,:)+deltau(1,ii)) -P(1,ii)*(Cv(ii,:)+deltav(1,ii)) zeros(1,N)];
    dual_history{ii} = zeros(48,k+1);
    charge = rand(N,1); discharge = rand(N,1);
    e = zeros(N,1); e(1,1) = Einit(ii) + P(1,ii)*deltaT*(rateu(1,ii)*charge(1,1) - ratev(1,ii)*discharge(1,1));
    for jj=2:N
        e(jj,1) = e(jj-1,1) + P(1,ii)*deltaT*(rateu(1,ii)*charge(jj,1) - ratev(1,ii)*discharge(jj,1));
    end
    x(:,ii) = [charge; discharge; e];
end
a = DoublyStochUndirected(I);
X_tl=[zeros(1,2*N) Emin*ones(1,N)];

%matrices for plotting and showing convergence properties
estimated_violation = zeros(48,k+1); better_violation = zeros(48,k+1); convIters = ones(1,I);
graphA = zeros(1,k+1); graphB = graphA; graphL = graphA;
graphAnorm = graphA; graphBnorm = graphA; graphLnorm = graphA;
graphE = zeros(I,k+1); graphF = zeros(I,k+1); graphG = zeros(I,k+1); graphJ = zeros(I,k+1);
distMatrixCreationTime = toc; %call this variable in command window to see actual time
if distMatrixCreationTime <= 60
    fprintf('Time taken to create matrices: %f seconds\n',toc)
else
    fprintf('Time taken to create matrices: %f minutes\n',toc/60)
end
distMatrixCreationTime = toc/60; %call this variable in command window to see actual time

%execute algorithm in series
fprintf('Now executing distributed algorithm\n')
tic
%shuffle this to get new sequence of agents (instead of going like agent 1, agent 2 ... at the start of each 
%iteration, you randomize the sequence)
shuffle_me = 1:I; 
T = 2; %after every T iterations (you can change this period), the adjacency matrix changes (time-varying graph)
foundIter = ones(1,I); %activate this for a node when its convergence error becomes small
%first get the iteration
for j=0:k
    actual_cost = 0; estimated_cost = 0; better_cost = 0;
    estimated_violation(:,j+1) = zeros(48,1); better_violation(:,j+1) = zeros(48,1);
    %shuffle your nodes every iteration to get a new sequence
    shuffle_me = shuffle_me(randperm(length(shuffle_me)));
    
    %time-varying graph, so compute new adjacency matrix periodically
    if j > 0 && rem(j,T)
        a = DoublyStochUndirected(I);
    end
    %go through each agent sequentially
    for z=1:I
    %index into the bigger matrices and cells to extract the matrices for
    %the current agent you are going to apply algorithm on
    r = shuffle_me(z); c_t=c(r,:); A_t=A{r}; a_t=a(r,:); x_in=x(:,r);
    Aineq=A_ineq{r}; bineq=b_ineq{r}; Aeq=A_eq{r}; beq=b_eq{r}; Xu=X_tu{r}; v_t = dual_history{r};
    mem_num_t=mem_num(:,r); mem_den_t=mem_den(1,r); %get memory elements here
    [x_t,x_hat,v_hat,e_n,mem_num_t,mem_den_t,final_cost] = node(c_t,beta,a_t,A_t,b,Aineq,bineq',Aeq,beq',v,x_in,j,mem_num_t,mem_den_t,X_tl,Xu,I);
    %update the global dual matrix. But we also need to plot the evolution
    %of each dual vector across the given number of iterations. So, the
    %evolution of each dual vector is nothing more than a 48 by k+1 matrix
    %and you index this matrix column-wise to store the dual vector you
    %compute at the given iteration. Then you index a cell according to the
    %current agent you were working on and store this 48 by k+1 matrix
    %inside there. So, for example, if you have 100 cars, that means
    %dual_history is a cell array of 100 matrices. If the total iterations
    %are 1000, then each matrix in this cell is 48 by 1001 (we go from iteration 0
    %to 1000) and every column represents the value of the dual vector for
    %that agent at a generic iteration between 0 and 1001.
    v(:,r) = v_hat; v_t(:,j+1) = v_hat; dual_history{r} = v_t; 
    %state vector of each agent stored in a similar fashion as the dual
    %vectors. x_t is the actual value of the state and x_hat is the NORMAL
    %estimated value of state
    x(:,r) = x_t; xEst(:,r) = x_hat;
    %compute dual cost for a given agent and add it to actual_cost. After
    %you have worked on all agents for a given iteration, your actual_cost
    %will have the total sum of all the dual cost values you calculated for
    %all the agents (same reasoning for estimated_cost)
    actual_cost = actual_cost + final_cost; estimated_cost = estimated_cost + (c_t*x_hat);
    mem_num(:,r) = mem_num_t; mem_den(1,r) = mem_den_t;      
    %start computing better estimate if convergence error is below a
    %threshold. Store the iteration at which threshold is violated inside
    %the convIters array. Because, once the convergence error is below a
    %threshold, it will stay below at every iteration. So, according to
    %this logic, each agent will keep entering this if-condition and
    %update the iteration at which the convergence error becomes less than
    %the threshold. But we want to get the iteration, at which convergence
    %error is below a threshold, only once when this happens and then we
    %just recalculate the BETTER estimate of the state from that iteration
    %at which the convergence error dropped below a threshold for the first
    %time.
    if e_n <= convergence_error && foundIter(1,r) == 1
        %Due to using a memory-based approach to computing estimate, all we
        %have to do is remove the memory (equate both numerator and denominator
        %memory to 0) which indicates to the agent that it should recompute
        %the estimate but this time from a new iteration (and not the very first
        %iteration). And, so your memory is rebuilt from this new %iteration
        foundIter(1,r) = 0; newmem_num(:,r) = zeros(3*N,1); newmem_den(1,r) = 0; convIters(1,r) = j+1;
    end
    newmem_num(:,r) = newmem_num(:,r) + (beta/(j+1))*x_t; newmem_den(1,r) = newmem_den(1,r) + (beta/(j+1));
    xBet(:,r) = newmem_num(:,r) / newmem_den(1,r); %the BETTER estimate
    %same reasoning here as used for actual_cost and estimated_cost
    estimated_violation(:,j+1) = estimated_violation(:,j+1) + A_t*x_hat; 
    better_violation(:,j+1) = better_violation(:,j+1) + A_t*xBet(:,r);
    better_cost = better_cost + (c_t*xBet(:,r));
    end
    %we have now worked on all agents for a given iteration
    %subtract the upper bound from the sum of all the agents violation
    better_violation(:,j+1) = better_violation(:,j+1) - b;
    estimated_violation(:,j+1) = estimated_violation(:,j+1) - b;
    fprintf('iteration %d complete\n',j)
    fprintf('Your dual cost is %f\n',actual_cost)
    fprintf('Your estimated cost is %f\n',estimated_cost)
    fprintf('Your better estimated cost is %f\n',better_cost)
    %plot the absolute difference between this cost and the cost calculated
    %using the centralized approach
    if noLinprog == 0
        graphA(1,j+1) = abs(actual_cost - cost); graphB(1,j+1) = abs(estimated_cost - cost); 
        graphL(1,j+1) = abs(better_cost - cost);
    else
        %if you are not comparing your solution to the centralized version
        %then just plot the cost as it is
        graphA(1,j+1) = actual_cost; graphB(1,j+1) = estimated_cost; graphL(1,j+1) = better_cost;
    end
    %useful to plot the cost as it is to show it is decreasing
    graphAnorm(1,j+1) = actual_cost; graphBnorm(1,j+1) = estimated_cost; graphLnorm(1,j+1) = better_cost;
    %store the dual variable vectors of all agents and their average too
    dual_history{j+1} = v; fullv = sum(v,2)/I;
    %show convergence using better estimate of the state
    xSum = sum(xBet,2)/I;
    for r=1:I
        graphE(r,j+1) = vecnorm(v(:,r)-fullv); %difference with average of dual approaches 0
        graphF(r,j+1) = vecnorm(x(:,r)-xSum);   %difference with average
        graphJ(r,j+1) = vecnorm(x(:,r)-x_star(:,r)); %difference of every node with true solution converges to 0
        if j-1>0
            graphG(r,j+1) = graphF(r,j+1)*(beta/(j+1)) + graphF(r,j); %summability condition
        end
    end
end
distSolveTime = toc;
if distSolveTime <= 60
    fprintf('Time taken: %f seconds\n',toc)
else
    fprintf('Time taken: %f minutes\n',toc/60)
end
distSolveTime = toc/60; %call this variable in command window to see actual time

%plot each graph manually in command window for better viewing
subplot(4,4,1); plot(graphA(1,:))
title('Relative Dual Cost')
xlabel('number of iterations') 
if noLinprog == 0
    ylabel('Dual - Centralized Cost')
else
    yLabel('Dual Cost')
end 
subplot(4,4,2); plot(graphB(1,:))
title('Relative Estimated Cost')
xlabel('number of iterations') 
if noLinprog == 0
    ylabel('Estimated - Centralized Cost')
else
    ylabel('Estimated Cost')
end
subplot(4,4,3); plot(graphL(1,:))
hold on
%show on graph the iterations at which convergence threshold was achieved
for ii=1:I
    plot(convIters(1,ii),graphL(1,convIters(1,ii)),'r*')
end
title('Relative Better Estimated Cost')
xlabel('number of iterations') 
if noLinprog == 0
    ylabel({'Better Estimated -';'Centralized Cost'})
else
    ylabel('Better Estimated Cost')
end
subplot(4,4,4); plot(graphBnorm(1,:))
title('Estimated Cost')
xlabel('number of iterations')
ylabel('Estimated Cost')
subplot(3,4,5); plot(graphLnorm(1,:))
title('Better Estimated Cost')
xlabel('number of iterations')
ylabel('Better Estimated Cost')
subplot(3,4,6); plot(graphE(1:I,1:k)')
title({'Evolution of average of';'all dual variable vectors'})
xlabel('number of iterations')
subplot(3,4,7); plot(graphF(1:I,1:k)')
title({'Evolution of norm of';'difference with average'})
xlabel('number of iterations')
subplot(3,4,8); plot(graphG(1:I,1:k)')
title({'Evolution of weighted norm';'of difference with average'})
xlabel('number of iterations')
subplot(3,4,9); plot(graphJ(1:I,1:k)')
xlabel('number of iterations')
title({'Evolution of norm of';'difference with true solution'})
subplot(3,4,10); plot(estimated_violation(1:48,1:k)')
xlabel('number of iterations')
title('Evolution of estimated violation')
subplot(3,4,11); plot(better_violation(1:48,1:k)')
xlabel('number of iterations')
title('Evolution of better violation')
subplot(3,4,12)
hold on
for zz=1:I
    fg=dual_history{zz}; plot(fg(1,:))
end
xlabel('number of iterations')
title('Evolution of dual vectors')
hold off

%plot this separately in command window to see duality gap
% plot(graphAnorm(1,:))
% title('Dual Cost')
% xlabel('number of iterations')
% ylabel('Dual Cost')
% hold on
% plot(graphAnorm(1,:))
% plot(graphBnorm(1,:))
% plot(graphLnorm(1,:))
% hold off
% title('Duality gap')
% xlabel('number of iterations')
% ylabel({'Dual vs Estimated';'vs Better Estimated'})