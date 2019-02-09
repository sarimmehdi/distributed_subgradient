/*
 * optimtask.h
 *
 *  Created on: Dec 22, 2018
 *      Author: sarim
 */

#ifndef OPTIMTASK_H_
#define OPTIMTASK_H_

class OptimTask
{
 public:
//sample data for EACH NODE (same variable names as in Matlab)
typedef struct myData
{
  myData(int totalAgents, int totalSubsystems) : I(totalAgents), N(totalSubsystems) { }
  int I;
  int N;
  double P;
  double Emin;
  double Emax;
  double Einit;
  double Eref;
  double rateu;
  double ratev;
  double deltaT;
  double Pmax;
  double Pmin;
  double deltau;
  double deltav;
  arma::mat Cu;
  arma::mat Cv;
} Data;

//same as in Matlab code
typedef struct myNode_Data
{
 myNode_Data(int totalAgents, int totalSubsystems, double rateB) : I(totalAgents), N(totalSubsystems), B(rateB) { }
 int I;
 int N;
 double B;
 double cost_true;                               //the dual cost
 double cost_estimate;                           //C*x where x is the estimated one (x_hat)
 double cost_better;                             //C*x where x is the better estimate (x_bet)
 double memory_den;
 double newmemory_den;
 arma::mat violation_true;       //A*x - b where x is the normal estimate (x_hat)
 arma::mat violation_estimate;   //A*x - b where x is the better estimate (x_bet)
 arma::mat memory_num;
 bool recomputeIter;
 arma::mat newmemory_num;
 arma::mat x_t;
 arma::mat x_hat;
 arma::mat x_bet;
 arma::mat X_tu;
 arma::mat X_tl;
 arma::mat C;
 arma::mat A;
 arma::mat b;
 arma::mat Aeq;
 arma::mat beq;
 arma::mat Aineq;
 arma::mat bineq;
 arma::mat v;
} Node_Data;

  OptimTask() { }
  ~OptimTask() { }
  void generate_node_data(Data &d, Node_Data &D, int world_rank);
  void generate_data(Data &d, int I, int N, int myRank);
  void distributed_algorithm(Node_Data &theData, int currentIteration, arma::mat &v_t, arma::mat &theRow, double error);
  void undirectedDoublyStochastic(int N, arma::mat &a);
  //(from matlab) x_new = linprog(f,Aineq,bineq,Aeq,beq,min,max,options);
  void linprog(arma::mat &bineq, arma::mat &f, arma::mat &Aineq, arma::mat &x_t,
		  arma::mat &Aeq, arma::mat &beq, arma::mat min, arma::mat max);
};




#endif /* OPTIMTASK_H_ */
