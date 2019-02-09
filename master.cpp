#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <mpi.h>
#include <glpk.h>
#include <ctime>
#include <armadillo>      //library for linear algebra in C++
#include "optimtask.h"    //data for the task (data generated and manipulated here)
#include "gnuplot-iostream.h"   //plot data using gnu-plot

using namespace std;
using namespace arma;

//uncomment if you want to see problem solved as undistributed (size of matrices is too huge, even at 50 vehicles. Recommended to keep this
//commented and just use result of centralized solution from Matlab for comparison)
//#define undistributed

int main(int argc, char** argv)
{
  int totalIteration = 1000; bool complete = false; clock_t timer;

  //initialize MPI environment
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  double trueCost = 0.0, estimateCost = 0.0, betterCost = 0.0; bool createData = false;
  mat trueViolation = zeros<mat>(48,1), estimateViolation = zeros<mat>(48,1), adjacency, yourRow = zeros<mat>(1,world_size-1);
  OptimTask myTask; OptimTask::myNode_Data D = {world_size-1, 24, 0.001}; OptimTask::myData d = {world_size-1, 24};
  int currentIteration = 0, period = 2; mat v_t = zeros<mat>(48,1);
  vector<double> yaxis1, yaxis2, yaxis5;
  vector<mat> yaxis3, yaxis4;

  //execute algorithm as undistributed using linprog
  #ifdef undistributed
  /*master first generates the data and sends it to all the nodes. Each node generates its matrices according to the data and sends the matrices
    back to the master for it to coalesce into larger matrices on which it applies linprog*/
  if (world_rank == world_size-1)
  {
	timer = clock();
	myTask.generate_data(d, world_size-1, 24, 0);
    mat c, A, A_eq, b_eq, A_ineq, b_ineq, X_tu, X_tl, b, x;
    c = zeros<mat>(1,3*d.N*d.I); A = zeros<mat>(2*d.N,3*d.N*d.I); b = zeros<mat>(2*d.N*d.I,1); A_eq = zeros<mat>(d.N*d.I,3*d.N*d.I);
    b_eq = zeros<mat>(d.N*d.I,1); b_ineq = zeros<mat>((d.N+1)*d.I,1); X_tu = zeros<mat>(1,3*d.N*d.I); X_tl = zeros<mat>(1,3*d.N*d.I);
    A_ineq = zeros<mat>((d.N+1)*d.I,3*d.N*d.I); x = zeros<mat>(3*d.N*d.I,1);
    //generate data and send it to each node (we still take advantage of parallelism, even though, in the end, the problem is solved centrally)
	for (int i = 0; i < d.I; i++)
	{
		MPI_Send(&d.N, 1, MPI_INT, i, 0, MPI_COMM_WORLD); MPI_Send(&d.P, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.Emin, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(&d.Emax, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.Einit, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(&d.Eref, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.rateu, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(&d.ratev, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.deltaT, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(&d.Pmax, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.Pmin, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(&d.deltau, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(&d.deltav, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); MPI_Send(d.Cu.begin(), d.Cu.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		MPI_Send(d.Cv.begin(), d.Cv.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
	}

	//receive matrices from each node
	D.C = zeros<mat>(1,3*d.N); D.A = zeros<mat>(2*d.N,3*d.N); D.b = zeros<mat>(2*d.N,1); D.bineq = zeros<mat>(d.N+1,1); D.Aineq = zeros<mat>(d.N+1,3*d.N);
	D.beq = zeros<mat>(d.N,1); D.Aeq = zeros<mat>(d.N,3*d.N); D.x_t = zeros<mat>(3*d.N,1); D.X_tu = zeros<mat>(1,3*d.N); D.X_tl = zeros<mat>(1,3*d.N);
	for (int i = 0; i < d.I; i++)
	{
		MPI_Recv(D.C.begin(), D.C.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.A.begin(), D.A.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.b.begin(), D.b.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.bineq.begin(), D.bineq.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.Aineq.begin(), D.Aineq.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.beq.begin(), D.beq.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.Aeq.begin(), D.Aeq.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.x_t.begin(), D.x_t.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.X_tu.begin(), D.X_tu.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D.X_tl.begin(), D.X_tl.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		c.submat(0,3*d.N*i,0,3*d.N*(i+1)-1) = D.C; A.submat(0,3*d.N*i,2*d.N-1,3*d.N*(i+1)-1) = D.A; b.submat(2*d.N*i,0,2*d.N*(i+1)-1,0) = D.b;
		A_ineq.submat((d.N+1)*i,3*d.N*i,(d.N+1)*i+d.N,3*d.N*(i+1)-1) = D.Aineq; b_ineq.submat((d.N+1)*i,0,(d.N+1)*i+d.N,0) = D.bineq;
		A_eq.submat(d.N*i,3*d.N*i,d.N*i+d.N-1,3*d.N*i+3*d.N-1) = D.Aeq; b_eq.submat(d.N*i,0,d.N*i+d.N-1,0) = D.beq;
		x.submat(3*d.N*i,0,3*d.N*i+3*d.N-1,0) = D.x_t; X_tu.submat(0,3*d.N*i,0,3*d.N*i+3*d.N-1) = D.X_tu;
		X_tl.submat(0,3*d.N*i,0,3*d.N*i+3*d.N-1) = D.X_tl;
	}

	mat finalAineq = join_cols(A,A_ineq), finalBineq = join_cols(b,b_ineq);

	cout << "GENERATING MATRICES TOOK " << double(clock() - timer) / (CLOCKS_PER_SEC * 60) << " MINUTES" << endl;
	cout << "ALL MATRICES GENERATED, NOW EXECUTING LINPROG..." << endl;
	timer = clock();
	myTask.linprog(finalBineq,c,finalAineq,x,A_eq,b_eq,X_tl,X_tu);
	cout << "SOLVING UNDISTRIBUTED VERSION OF PROBLEM TOOK " << double(clock() - timer) / (CLOCKS_PER_SEC * 60) << " MINUTES" << endl;
	cout << "THE COST IS " << c[c.size()-1]*x[x.size()-1] << endl;
  }

  //nodes generate matrices and send them to master for the final centralized problem construction
  else
  {
	  //receive data from master
	  MPI_Recv(&d.N, 1, MPI_INT, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.P, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Emin, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Emax, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Einit, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Eref, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.rateu, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.ratev, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.deltaT, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Pmax, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.Pmin, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.deltau, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&d.deltav, 1, MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  d.Cu = zeros<mat>(1,d.N); d.Cv = zeros<mat>(1,d.N);
	  MPI_Recv(d.Cu.begin(), d.Cu.size(), MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(d.Cv.begin(), d.Cv.size(), MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  myTask.generate_node_data(d, D, world_rank); //generate matrices

	  //send matrices to master
	  MPI_Send(D.C.begin(), D.C.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.A.begin(), D.A.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.b.begin(), D.b.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.bineq.begin(), D.bineq.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.Aineq.begin(), D.Aineq.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.beq.begin(), D.beq.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.Aeq.begin(), D.Aeq.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.x_t.begin(), D.x_t.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.X_tu.begin(), D.X_tu.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
	  MPI_Send(D.X_tl.begin(), D.X_tl.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  /*remember that total number of MPI nodes is 1 greater than the total number of agents in distributed network (master only synchronizes data
   * at the end of each iteration)*/
  timer = clock();
  while(!complete)
  {
	  /*master generates adjacency matrix periodically and updates the global dual matrix after all nodes have executed the algorithm for the
	    given iteration. It is also responsible for plotting the graph*/
	  if (world_rank == world_size-1)
	  {
		  //master generates adjacency matrix and sends each row to each agent according to period chosen by user
		  if(currentIteration % period == 0)
		  {
			myTask.undirectedDoublyStochastic(world_rank, adjacency);
			//generate data, then create matrices for each node and send them
			for (int i = 0; i < d.I; i++)
			{
				yourRow = adjacency.row(i);
				MPI_Send(yourRow.begin(), yourRow.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		  }

		  /*master pieces together a new global dual variable matrix after all nodes have completed one iteration and sent over their respective
		    column of the global dual variables matrix*/
		  D.v = zeros<mat>(48,d.I); D.violation_true = zeros<mat>(48,1); D.violation_estimate = zeros<mat>(48,1);
		  for (int i = 0; i < d.I; i++)
		  {
			 MPI_Recv(v_t.begin(), v_t.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); D.v.col(i) = v_t;
		     MPI_Recv(&D.cost_true, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(&D.cost_estimate, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(&D.cost_better, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(D.violation_true.begin(), D.violation_true.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(D.violation_estimate.begin(), D.violation_estimate.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     trueCost += D.cost_true; estimateCost += D.cost_estimate; betterCost += D.cost_better;
		     trueViolation += D.violation_true; estimateViolation += D.violation_estimate;
		  }

		  //Note: trying to print arma::mat using gnuplot was unsuccessful (unable to get exact same plot as matlab with entire matrix on one plot)
		  //However, with this C++ implementation we can faithfully show the duality gap closing, which is also a good simulation to show!
		  //print message after all nodes have completed an iteration
		  cout << "ITERATION " << currentIteration << " COMPLETE" << endl;
		  cout << "TRUE COST IS " << trueCost << endl; cout << "ESTIMATED COST IS " << estimateCost << endl;
		  cout << "BETTER ESTIMATED COST IS " << betterCost << endl;

		  //save data in vectors (multiplied by 1000 for easy plot visualization) to be later plotted
		  yaxis3.push_back(trueViolation * 1000); yaxis4.push_back(estimateViolation * 1000);
		  yaxis1.push_back(trueCost * 1000); yaxis2.push_back(estimateCost * 1000); yaxis5.push_back(betterCost * 1000);

		  trueCost = 0.0; estimateCost = 0.0; betterCost = 0.0; trueViolation = zeros<mat>(48,1); estimateViolation = zeros<mat>(48,1);

		  for (int i = 0; i < d.I; i++) { MPI_Send(D.v.begin(), D.v.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD); }

		  //wait for all tasks to update their iteration before updating master's iteration
		  for (int i = 0; i < d.I; i++)
		  {
			  MPI_Recv(&currentIteration, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  }
		  if (currentIteration > totalIteration) { complete = true; }
	  }

	  //your nodes that execute distributed algorithm
	  else
	  {
		  //first each node generates all of its data (excluding the adjacency matrix, which it will receive from master)
		  if(currentIteration % period == 0)
		  {
			  MPI_Recv(yourRow.begin(), yourRow.size(), MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  }

		  //generate data once in the beginning
		  if (!createData)
		  {
			  createData = true; myTask.generate_data(d, world_size-1, 24, world_rank); myTask.generate_node_data(d, D, world_rank);
		  }

		  //main if condition where all the magic happens
		  if (currentIteration <= totalIteration)
		  {
			  myTask.distributed_algorithm(D, currentIteration, v_t, yourRow, 0.001);
			  MPI_Send(v_t.begin(), v_t.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Send(&D.cost_true, 1, MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Send(&D.cost_estimate, 1, MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Send(&D.cost_better, 1, MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Send(D.violation_true.begin(), D.violation_true.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Send(D.violation_estimate.begin(), D.violation_estimate.size(), MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD);
			  MPI_Recv(D.v.begin(), D.v.size(), MPI_DOUBLE, world_size-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  }
		  currentIteration++; MPI_Send(&currentIteration, 1, MPI_INT, world_size-1, 0, MPI_COMM_WORLD);
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
  }

  //plot graph here
  if (world_rank == world_size-1)
  {
	  cout << "ALGORITHM HAS COMPLETED EXECUTION!" << endl;
	  cout << "SOLVING DISTRIBUTED ALGORITHM TOOK " << double(clock() - timer) / (CLOCKS_PER_SEC * 60) << " MINUTES" << endl;
	  Gnuplot gp1, gp2, gp3;
	  gp1 << "plot" << gp1.file1d(yaxis1) << "with lines title 'dual'" << endl;
	  gp2 << "plot" << gp2.file1d(yaxis2) << "with lines title 'estimate'" << endl;
	  gp3 << "plot" << gp3.file1d(yaxis5) << "with lines title 'better'" << endl;
  }
  MPI_Finalize();
  return 0;
}
