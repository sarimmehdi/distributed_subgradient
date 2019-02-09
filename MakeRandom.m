function [I,P,Emin,Emax,Einit,Eref,rateu,ratev,deltaT,N,Pmax,Pmin,Cu,Cv,deltau,deltav] = MakeRandom(i,n,bias)
I = i;
%total cars must be divided by number that gives 0 remainder
while rem(I,n) ~= 0
    n=n+1;
end
%assign significantly larger power values to some (or more) of the cars if required
if n ~= 0
    if bias == 1
        PHigh = (50-30).*rand(1,(I-I/n)) + 5; PNormal = (5-3).*rand(1,I/n) + 3;
        EmaxHigh = (100-80).*rand(1,(I-I/n)) + 20; EmaxNormal = (16-8).*rand(1,I/n) + 8;
    else
        PHigh = (50-30).*rand(1,I/n) + 5; PNormal = (5-3).*rand(1,(I-I/n)) + 3;
        EmaxHigh = (150-120).*rand(1,I/n) + 20; EmaxNormal = (16-8).*rand(1,(I-I/n)) + 8;
    end

    P = [PHigh PNormal]; Emax = [EmaxHigh EmaxNormal];
else
    P = (5-3).*rand(1,I) + 3; Emax = (16-8).*rand(1,I) + 8;
end
Emin = 1; EmaxTemp = (16-8).*rand(1,I) + 8;
Einit = ((0.5-0.2).*rand(1,I) + 0.2).*EmaxTemp;
Eref = ((0.8-0.55).*rand(1,I) + 0.55).*Emax;
rateu = 1-((0.075-0.015).*rand(1,I) + 0.015);
ratev = 1+((0.075-0.015).*rand(1,I) + 0.015);
deltaT = 0.33;
N = 24;
Pmax = 3*I;
Pmin = -Pmax;
Cu = ((35-19).*rand(I,N) + 19)*0.001;
Cv = 1.1*Cu;
deltau = (((0.3+0.3).*rand(1,I) - 0.3)) * 0.001;
deltav = (((0.3+0.3).*rand(1,I) - 0.3)) * 0.001;
end