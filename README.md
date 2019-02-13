# distributed_subgradient
Distributed Dual Subgradient Scheme for Charging of Plug-in Electric Vehicles

FOR MATLAB:
Simple copy and paste all the .m files in one place. Then, open runme_series.m and click Run

FOR C++:
Copy and paste all the files in one place. Open a terminal and go to where you copy pasted the stuff. Write the following to compile:
mpic++ -o "Optional" master.cpp optimtask.cpp -larmadillo -lboost_iostreams -lboost_system -lboost_filesystem -lglpk

(Make sure gnuplot, boost, armadillo and glpk are installed on your system)

Then, write the following to run (If you want to see execution for 100 agents, the argument must be 101 and not 100):
mpirun -n 101 ./Optional
