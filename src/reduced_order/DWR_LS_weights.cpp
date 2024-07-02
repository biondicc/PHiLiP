#include "DWR_LS_weights.h"
#include <iostream>

#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include "linear_solver/NNLS_solver.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
DWRLSWeights<dim,nstate>::DWRLSWeights(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input,
    const Epetra_CrsMatrix &A, 
    Epetra_MpiComm &Comm, 
    Epetra_Vector &b,
    dealii::Vector<double> DWR_weights_mesh):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    C(A),
    Comm_(Comm),
    d(b),
    DWR_reduced_mesh(DWR_weights_mesh),
    weights(A.ColMap()) 
{
    
}

template <int dim, int nstate>
void DWRLSWeights<dim,nstate>::PositiveSetMatrix(Epetra_CrsMatrix &P_mat){
    // Create matrix P_mat which contains the positive set of columns in A

    // Create map between indeweightsset and the columns to be added to P_mat
    std::vector<int> colMap(C.NumGlobalCols());
    int numCol = 0;
    for(int j = 0; j < C.NumGlobalCols(); j++){
        if (DWR_reduced_mesh[j] != 0) {
            colMap[j] = numCol;
            numCol++;
        }
    }

    // Fill Epetra_CrsMatrix P_mat with columns of A in set P
    for(int i =0; i < C.NumGlobalRows(); i++){
        double *row = new double[C.NumGlobalCols()];
        int numE;
        const int globalRow = C.GRID(i);
        C.ExtractGlobalRowCopy(globalRow, C.NumGlobalCols(), numE , row);
        for(int j = 0; j < C.NumGlobalCols(); j++){
            if (DWR_reduced_mesh[j] != 0) {
                P_mat.InsertGlobalValues(i, 1, &row[j] , &colMap[j]);
            }
        }
    }
}

template <int dim, int nstate>
void DWRLSWeights<dim,nstate>::SubIntoX(Epetra_Vector &temp){
  // Substitute new values into the solution vector
  std::vector<int> colMap(C.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < weights.GlobalLength(); j++){
    if (DWR_reduced_mesh[j] != 0) {
      colMap[j] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < weights.GlobalLength(); j++){
    if (DWR_reduced_mesh[j] != 0) {
      weights[j] = temp[colMap[j]];
    }
  }
}

template <int dim, int nstate>
void DWRLSWeights<dim,nstate>::solve(){
    int num_elements_reduced_mesh = 0;
    for(int j = 0; j < weights.GlobalLength(); j++){
        if (DWR_reduced_mesh[j] != 0) {
            num_elements_reduced_mesh++;
        }
    }

    std::cout << "Num non-zeros"<< std::endl;
    std::cout << num_elements_reduced_mesh << std::endl;

    // Create matrix P_mat with columns from set P
    Epetra_Map Map(C.NumGlobalRows(),0,Comm_);
    Epetra_Map ColMap(num_elements_reduced_mesh,0,Comm_);
    Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, num_elements_reduced_mesh);
    PositiveSetMatrix(P_mat);
    P_mat.FillComplete(ColMap, Map);

    // Create temporary solution vector temp which is only the length of numInactive
    Epetra_Vector temp(P_mat.ColMap());

    // Set up normal equations
    Epetra_CrsMatrix PtP(Epetra_DataAccess::View, P_mat.ColMap(), P_mat.NumMyCols());
    EpetraExt::MatrixMatrix::Multiply(P_mat, true, P_mat, false, PtP);

    Epetra_Vector Ptd (P_mat.ColMap());
    P_mat.Multiply(true, d, Ptd);

    // Solve least-squares problem in inactive set only
    Epetra_LinearProblem problem(&PtP, &temp, &Ptd);

    // Direct Solver Setup
    Amesos Factory;
    std::string SolverType = "Klu";
    std::unique_ptr<Amesos_BaseSolver> Solver(Factory.Create(SolverType, problem));

    Teuchos::ParameterList List;
    List.set("OutputLevel", 2);
    Solver->SetParameters(List);
    Solver->SymbolicFactorization();
    Solver->NumericFactorization();
    Solver->Solve();


/*     Epetra_LinearProblem problem(&P_mat, &temp, &d);
    AztecOO solver(problem);

    // Iterative Solver Setup
    solver.SetAztecOption(AZ_conv, AZ_rhs);
    solver.SetAztecOption( AZ_precond, AZ_Jacobi);
    solver.SetAztecOption(AZ_output, AZ_none);
    solver.Iterate(1000, 1E-8);

    SubIntoX(temp); */

/*     NNLS_solver NNLS_prob(this->all_parameters, this->parameter_handler, P_mat, Comm_, d, true);
    std::cout << "Solve NNLS problem..."<< std::endl;
    bool exit_con = NNLS_prob.solve();
    std::cout << exit_con << std::endl;
    temp = NNLS_prob.getSolution();
    std::cout << temp << std::endl; */
    SubIntoX(temp);
}



#if PHILIP_DIM==1
    template class DWRLSWeights<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
    template class DWRLSWeights<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // HyperReduction namespace
} // PHiLiP namespace