#ifndef __OUTPUT_VTK_ECSW_WEIGHTS_H__
#define __OUTPUT_VTK_ECSW_WEIGHTS_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Output the ECSW weights in a VTK file so they can be plotted in ParaView
/// NOTE: The folder the test reads from should have only one file beginning with "weights" which contains the last ECSW weights
/// found in the adaptive sampling procedure.
template <int dim, int nstate>
class OutputVTKWeights: public TestsBase
{
public:
    /// Constructor.
    OutputVTKWeights(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Read ECSW weights from the text file 
    bool getWeightsFromFile() const;

    /// Evaluate and output the "true" error at ROM Points
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of error sampling points
    mutable MatrixXd rom_points;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
