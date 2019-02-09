function WW = DoublyStochUndirected(N)
rng('shuffle');
while 1
    Adj = binornd(1,0.3,N,N); I_NN = eye(N,N); notI = ~I_NN; Adj = Adj.*notI;

    Adj = or(Adj,Adj'); test = (I_NN+Adj)^N;
    if ~any(any(~test))
        break
    end
end
DEGREE = sum(Adj); WW = zeros(N,N);

for ii = 1:N
    N_ii = find(Adj(:,ii) == 1)';
    for jj = N_ii
        WW(ii,jj) = 1/(1 + max(DEGREE(ii), DEGREE(jj)));
    end
    WW(ii,ii) = 1 - sum(WW(ii,:));
end