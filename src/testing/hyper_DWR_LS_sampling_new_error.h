#ifndef __HYPER_DWR_LS_SAMPLING_NEW_ERROR_H__
#define __HYPER_DWR_LS_SAMPLING_NEW_ERROR_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Runs hyperreduced adaptive sampling procedure 
template <int dim, int nstate>
class HyperDWRLSSamplingNewError: public TestsBase
{
public:
    /// Constructor.
    HyperDWRLSSamplingNewError(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Run hyperreduced adaptive sampling procedure 
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
