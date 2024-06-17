#ifndef __ADJOINT_BASED_WEIGHTS__
#define __ADJOINT_BASED_WEIGHTS__

#include <eigen/Eigen/Dense>
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
class AdjointWeights
{
public:
    /// Constructor
    AdjointWeights(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        MatrixXd snapshot_parameters_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type);

    /// Destructor
    ~AdjointWeights () {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// dg
    std::shared_ptr<DGBase<dim,double>> dg;

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    const MPI_Comm mpi_communicator; ///< MPI communicator.

    /// ODE Solve Type/ Projection Type (galerkin or petrov-galerkin)
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type;

    /// Weights for reduce mesh set
    dealii::Vector<double> weights;

    /// Reinitialize parameters
    Parameters::AllParameters reinitParams(const RowVectorXd& parameter) const;

    /// Fill entries of A and b
    void build_problem();
};

} // HyperReduction namespace
} // PHiLiP namespace

#endif