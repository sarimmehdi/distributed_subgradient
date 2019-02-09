#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <armadillo>
#include <glpk.h>
#include <algorithm>
#include "optimtask.h"

using namespace std;
using namespace arma;

//create a random doubly stochastic undirected adjacency matrix
void OptimTask::undirectedDoublyStochastic(int N, mat &a)
{
	//generate N by N matrix where each element is randomly picked from a binomial distribution with 1 trial and success probability of 0.3
	default_random_engine engine; binomial_distribution<int> distr(1,0.3); mat Adj(N,N); a = zeros<mat>(N,N);

    //Main diagonal is kept 0 in beginning
    Adj.imbue( [&]() { return distr(engine); } ); Adj.diag().zeros(); Adj = trimatl(Adj); Adj += Adj.t();

    /* Sum all the elements in each column of Adj and store in 1 by N matrix called degree. For a given row of a (a is the matrix I am trying to
     * convert to doubly stochastic), iterate through each column. For a given column j in the row i, check whether you have a 1 in the original
     * Adj matrix at the exact position. Then, at that position in a, put the value 1/max(temp1,temp2(j)). Example:
     * Let Adj = [0 1 1 1 1], now sum all the numbers of each column, degree = [4 4 3 3 4]
     *           [1 0 1 1 1]  Let's start with row 0 (first row) of a. We see that row 0 of Adj has 1 everywhere except at column 0 (first column)
     *           [1 1 0 0 1]  So, at column 0, we put 0 in a. At column 1, we check which is bigger: 0th element of degree (degree vector indexed
     *           [1 1 0 0 1]  according to the row of Adj you are currently searching) vector is 4 and 1st element of degree vector is 4. So, at
     *           [1 1 1 1 0]  position a(0,1) you put 1/4. So, at each position you first check which is bigger: The element in degree vector
     *                        obtained by indexing it according to the current row of Adj or the element in degree vector obtained by indexing
     *                        it according to the current column of the specific row which contains a 1 in the original Adj matrix. The bigger of
     *                        the two has its inverse taken and put in the corresponding place in Adj*/

    //make randomly generated undirected matrix doubly stochastic
    mat degree = sum(Adj); double compareRow, compareCol; mat temp; uword theN = N;
    for (uword i = 0; i < theN; i++)
    {
        compareRow = as_scalar(degree(i));
        for (uword j = 0; j < theN; j++)
        {
        	if(Adj(i,j) == 1) { compareCol = degree(j); a(i,j) = 1/max(compareRow,compareCol); }
        }
        temp = a.row(i); a(i,i) = 1 - accu(temp);
    }
}

//generate random data (data generated for one node)
void OptimTask::generate_data(Data &d, int I, int N, int myRank)
{
  d.I = I; d.N = N; d.Emin = 1; d.deltaT = 0.33; d.Pmax = 3 * I; d.Pmin = -3 * I; default_random_engine re(random_device{}());
  uniform_real_distribution<double> unifP(3,5); d.P = unifP(re); uniform_real_distribution<double> unifEmax(8,16); d.Emax = unifEmax(re);
  uniform_real_distribution<double> unifEinit(0.2,0.5); d.Einit = unifEinit(re); d.Einit = d.Einit*d.Emax;
  uniform_real_distribution<double> unifEref(0.55,0.8); d.Eref = unifEref(re); d.Eref = d.Eref*d.Emax;
  uniform_real_distribution<double> unifdeltau(-0.3,0.3); d.deltau = unifdeltau(re); d.deltau = d.deltau*0.001;
  uniform_real_distribution<double> unifdeltav(-0.3,0.3); d.deltav = unifdeltav(re); d.deltav = d.deltav*0.001;
  uniform_real_distribution<double> unifrate(0.015,0.075); double rate = unifrate(re); d.rateu = 1 - rate; d.ratev = 1 + rate;
  arma_rng::set_seed(myRank); d.Cu = (((randu<mat>(1,N))*16)+19)*0.001; d.Cv = d.Cu*1.1;
}

//create matrices to be sent to each node. Each Data has parameters generated for that specific node
void OptimTask::generate_node_data(Data &d, Node_Data &D, int world_rank)
{
  D.memory_den = 0.0; D.newmemory_den = 0.0; D.N = d.N; D.I = d.I; D.cost_true = 0.0, D.cost_estimate = 0.0; D.cost_better = 0.0;
  D.violation_true = zeros<mat>(48,1); D.violation_estimate = zeros<mat>(48,1); D.recomputeIter = false;

  /*From matlab:
    charge = rand(N,1); discharge = rand(N,1);
    e = zeros(N,1); e(1,1) = Einit(ii) + P(1,ii)*deltaT*(rateu(1,ii)*charge(1,1) - ratev(1,ii)*discharge(1,1));
    for jj=2:N
        e(jj,1) = e(jj-1,1) + P(1,ii)*deltaT*(rateu(1,ii)*charge(jj,1) - ratev(1,ii)*discharge(jj,1));
    end
    x(:,ii) = [charge; discharge; e];
  */
  arma_rng::set_seed(world_rank); mat e(d.N, 1, fill::zeros); D.v = zeros<mat>(48,d.I);
  mat charge(d.N, 1, arma::fill::randu); mat discharge(d.N, 1, arma::fill::randu);
  e(0,0) = d.Einit + d.P*d.deltaT*(d.rateu*charge(0,0) - d.ratev*discharge(0,0));
  for (uword i = 1; i < e.n_rows; i++) { e(i,0) = e(i-1,0) + d.P*d.deltaT*(d.rateu*charge(i,0) - d.ratev*discharge(i,0)); }
  D.x_t = join_cols(join_cols(charge, discharge),e);

  //(from matlab) mem_num=zeros(3*N,I);
  D.memory_num = zeros<mat>(3*d.N,1); D.newmemory_num = zeros<mat>(3*d.N,1);

  //(From matlab) b = [Pmax*ones(N,1); -Pmin*ones(N,1)];
  D.b = join_cols(d.Pmax*ones<mat>(d.N,1),-1*d.Pmin*ones<mat>(d.N,1));

  //(From matlab) mata = P(1,ii)*eye(N,N); matb = -P(1,ii)*eye(N,N); A{ii} = [mata matb zeros(N,N); matb mata zeros(N,N)];
  mat mata = d.P*eye<mat>(d.N,d.N), matb = -1*d.P*eye<mat>(d.N,d.N);
  D.A = join_cols(join_rows(join_rows(mata,matb),zeros<mat>(d.N,d.N)), join_rows(join_rows(matb,mata),zeros<mat>(d.N,d.N)));

  //(From matlab) tempMat1 = [eye(N,N); zeros(1,N)]; tempMat2 = zeros(N+1,N);
  //tempMat2(N+1,N) = -1; A_ineq{ii} = [tempMat1 tempMat1 tempMat2]; b_ineq{ii} = [ones(N,1); -Eref(1,ii)];
  mat tempMat1 = join_cols(eye<mat>(d.N,d.N),zeros<mat>(1,d.N)), tempMat2 = zeros<mat>(d.N+1,d.N); tempMat2(d.N,d.N-1) = -1;
  D.Aineq = join_rows(join_rows(tempMat1,tempMat1),tempMat2); D.bineq = join_cols(ones<mat>(d.N,1),-1*d.Eref*ones<mat>(1,1));

  //(From matlab) matA = tril(-P(1,ii)*deltaT*rateu(1,ii)*ones(N,N)); matB = tril(P(1,ii)*deltaT*ratev(1,ii)*ones(N,N));
  //A_eq{ii} = [matA matB eye(N,N)]; b_eq{ii} = Einit(1,ii)*ones(N,1);
  mat matA = trimatl(-1*d.P*d.deltaT*d.rateu*ones<mat>(d.N,d.N)), matB = trimatl(d.P*d.deltaT*d.ratev*ones<mat>(d.N,d.N));
  D.Aeq = join_rows(join_rows(matA,matB),eye<mat>(d.N,d.N)); D.beq = d.Einit*ones<mat>(d.N,1);

  //(From matlab) X_tl=[zeros(1,2*N) Emin*ones(1,N)];
  D.X_tl = join_rows(zeros<mat>(1,2*d.N),d.Emin*ones<mat>(1,d.N));

  //(From matlab) X_tu{ii} = [ones(1,2*N) Emax(1,ii)*ones(1,N)];
  D.X_tu = join_rows(ones<mat>(1,2*d.N),d.Emax*ones<mat>(1,d.N));

  //(From matlab) c(ii,:) = [P(1,ii)*(Cu(ii,:)+deltau(1,ii)) -P(1,ii)*(Cv(ii,:)+deltav(1,ii)) zeros(1,N)];
  D.C = join_rows(join_rows(d.P*(d.Cu+d.deltau),-1*d.P*(d.Cv+d.deltav)),zeros<mat>(1,d.N));
}

void OptimTask::distributed_algorithm(Node_Data &theData, int currentIteration, arma::mat &v_t, arma::mat &theRow, double error)
{
  //compute l by multiplying a column of v with the corresponding element of a and then adding up all the new columns into one single column
  mat l = zeros<mat>(theData.v.n_rows,1);
  for (uword i = 0; i < theData.v.n_cols; i++) { l += as_scalar(theRow(0,i)) * theData.v.col(i); }

  //Solve f = C + l'*A to find the coefficients of objective function which you will minimize using GLPK
  mat f = theData.C + l.t()*theData.A;

  //calculate dual cost: C*xInput + l'*A*xInput - l'*(b/I)
  theData.cost_true = as_scalar(theData.C*theData.x_t + l.t()*theData.A*theData.x_t - l.t()*(theData.b/theData.I));

  //Solve linear programming problem using GLPK
  linprog(theData.bineq, f, theData.Aineq, theData.x_t, theData.Aeq, theData.beq, theData.X_tl, theData.X_tu);

  //calculate gradient step-size
  double c = theData.B / (currentIteration+1);

  //get value of v vector for next iteration => (Matlab version) v_hat = l + c*A*x_new - c*(b/I); v_t(v_t<=0) = 0
  v_t = l + c*(theData.A*theData.x_t) - c*(theData.b/theData.I);
  v_t.elem( find(v_t < 0) ).zeros();

  /*if convergence error is below a user specified threshold then recompute estimate of the state from the iteration at which convergence error
    drops below that threshold*/
  double convError = norm(v_t - l);
  if (convError <= error && theData.recomputeIter == false)
  {
	  theData.recomputeIter = true; theData.newmemory_den = 0; theData.newmemory_num = zeros<mat>(72,1);
  }

  //calculate a better estimate of the node's state for the next iteration to show convergence
  theData.newmemory_num += (c*theData.x_t); theData.newmemory_den += c;
  theData.x_bet = theData.newmemory_num/theData.newmemory_den;

  //compute cost and violation for better estimated value of x
  theData.cost_better = as_scalar(theData.C * theData.x_bet);
  theData.violation_estimate = theData.A*theData.x_bet - (theData.b/theData.I);

  //calculate an estimate of the node's state for the next iteration to show convergence
  theData.memory_num += (c*theData.x_t); theData.memory_den += c;
  theData.x_hat = theData.memory_num/theData.memory_den;

  //compute cost and violation for estimated value of x
  theData.cost_estimate = as_scalar(theData.C * theData.x_hat);
  theData.violation_true = theData.A*theData.x_hat - (theData.b/theData.I);
}

//Matlab's linprog implemented using GLPK
void OptimTask::linprog(arma::mat &bineq, arma::mat &f, arma::mat &Aineq, arma::mat &x_t,
		  arma::mat &Aeq, arma::mat &beq, arma::mat min, arma::mat max)
{
  glp_term_out(GLP_OFF); glp_prob *lp; int ia[4000], ja[4000]; double ar[4000];
  lp = glp_create_prob(); glp_set_prob_name(lp, "subgradient"); glp_set_obj_dir(lp, GLP_MIN);

  //setup the bineq and beq
  glp_add_rows(lp, as_scalar(bineq.n_rows)+as_scalar(beq.n_rows)); string bineq_name; int r = 1; const char * str2c;
  for (uword i = 0; i < bineq.n_rows; i++)
  {
    bineq_name = "inequality " + to_string(r); str2c = bineq_name.c_str(); glp_set_row_name(lp, r, str2c);
    glp_set_row_bnds(lp, r, GLP_UP, 0.0, as_scalar(bineq(i,0))); r++;
  }

  string beq_name;
  for (uword i = 0; i < beq.n_rows; i++)
  {
    //set the name and upper bound
    beq_name = "equality " + to_string(r); str2c = beq_name.c_str(); glp_set_row_name(lp, r, str2c);
    glp_set_row_bnds(lp, r, GLP_FX, as_scalar(beq(i,0)), as_scalar(beq(i,0))); r++;
  }

  //setup the x vector where number of rows same as the total number of subsystems
  glp_add_cols(lp, x_t.n_rows); string x_name; r = 1;
  for (uword i = 0; i < x_t.n_rows; i++)
  {
    //first third of rows are charging times, next third are discharging times and last third are energy levels
    if (i < x_t.n_rows/3) { x_name = "u[" + to_string(r) + "]"; }
    else if (i >= x_t.n_rows/3 && i < x_t.n_rows*2/3) { x_name = "v[" + to_string(r) + "]"; }
    else { x_name = "e[" + to_string(r) + "]"; }
    str2c = x_name.c_str();

    //set the name, upper and lower bound, and coefficient
    glp_set_col_name(lp, r, str2c); glp_set_col_bnds(lp, r, GLP_DB, as_scalar(min(0,i)), as_scalar(max(0,i)));
    glp_set_obj_coef(lp, r, as_scalar(f(0,i))); r++;
  }

  /*Setup Aineq here. For example: ia[1] = 1, ja[1] = 1, ar[1] = 1.0 means the first (hence 1 as the index of all three arrays) element at
    location (1,1) has value 1.0 in the Aineq matrix t represents the total number of elements in your Aineq and Aeq matrix. r is the row of
    Aineq/ Aeq that you are currently scanning. w represents the current column of the current row you are in. All numbering must start from 1 and
    not 0. So, after you scan a row, you set w=1 because you will be at the first column of the next row. t will continue getting incremented*/
  int t = 0, w = 1; r = 1;
  for (uword i = 0; i < Aineq.n_rows; i++)
  {
    /*get the row of Aineq. Then, store the row, column, and value of every element in the Aineq matrix. Members of Aineq matrix are numbered
      from left to right and then down. So, if your Aineq matrix has two rows and three columns, element at 1,1 is 1, at 1,2 is 2, at 1,3 is
      3, at 2,1 is 4, at 2,2 is 5 and at 2,3 is 6*/
    for(uword j = 0; j < Aineq.n_cols; j++) { t++; ia[t] = r; ja[t] = w; ar[t] = as_scalar(Aineq(i,j)); w++; }
    r++; w = 1;
  }

  //Setup Aeq here using same method
  for (uword i = 0; i < Aeq.n_rows; i++)
  {
    for(uword j = 0; j < Aeq.n_cols; j++) { t++; ia[t] = r; ja[t] = w; ar[t] = as_scalar(Aeq(i,j)); w++; }
    r++; w = 1;
  }

  //Solve linear problem
  glp_load_matrix(lp, t, ia, ja, ar); glp_simplex(lp, NULL);

  //uncomment if you would also like to see the minimum value of the objective function
  //int z = glp_get_obj_val(lp);

  //get value of x vector for next iteration
  r = 1;
  for (uword i = 0; i < x_t.n_rows; i++)
  {
    //don't forget to round up or down before storing
	x_t(i,0) = glp_get_col_prim(lp, r); r++;
  }
  glp_delete_prob(lp);
}
