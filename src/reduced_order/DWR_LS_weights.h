#ifndef __DWR_LS_WEIGHTS__
#define __DWR_LS_WEIGHTS__

#include <eigen/Eigen/Dense>
#include <Epetra_MpiComm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include "dg/dg_base.hpp"
#include "pod_basis_base.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

template <int dim, int nstate>
class DWRLSWeights
{
public:
    /// Constructor
    DWRLSWeights(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const Epetra_CrsMatrix &A, 
        Epetra_MpiComm &Comm, 
        Epetra_Vector &b,
        dealii::Vector<double> DWR_weights_mesh);

    /// Destructor
    virtual ~DWRLSWeights () {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix for the NNLS Problem
    const Epetra_CrsMatrix C;
    
    /// Epetra Comm
    Epetra_MpiComm Comm_;

    /// RHS Vector for the NNLS Problem
    Epetra_Vector d;

    /// Reduced Mesh Set identified by DWR Error at cells
    dealii::Vector<double> DWR_reduced_mesh;
    
    /// Weights
    Epetra_Vector weights;

    /// Creates a matrix using the columns in A in the set P
    void PositiveSetMatrix(Epetra_CrsMatrix &P_mat);

    /// Replaces the entries with x with the values in temp
    void SubIntoX(Epetra_Vector &temp);

    /// Call to solve NNLS problem
    void solve();
};

} // HyperReduction namespace
} // PHiLiP namespace

#endif