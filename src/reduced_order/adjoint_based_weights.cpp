#include "adjoint_based_weights.h"
#include <iostream>

#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "linear_solver/linear_solver.h"
#include "flow_solver/flow_solver_factory.h"

#include "mesh/mesh_adaptation/mesh_error_estimate.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
AdjointWeights<dim,nstate>::AdjointWeights(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input,
    std::shared_ptr<DGBase<dim,double>> &dg_input, 
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
    MatrixXd snapshot_parameters_input,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , dg(dg_input)
        , pod(pod)
        , snapshot_parameters(snapshot_parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , ode_solver_type(ode_solver_type)
{
}

template <int dim, int nstate>
Parameters::AllParameters AdjointWeights<dim, nstate>::reinitParams(const RowVectorXd& parameter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a"){
                parameters.burgers_param.rewienski_a = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b"){
                parameters.burgers_param.rewienski_b = parameter(0);
            }
        }
        else{
            parameters.burgers_param.rewienski_a = parameter(0);
            parameters.burgers_param.rewienski_b = parameter(1);
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
                parameters.euler_param.angle_of_attack = parameter(0); //radians!
            }
        }
        else{
            parameters.euler_param.mach_inf = parameter(0);
            parameters.euler_param.angle_of_attack = parameter(1); //radians!
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
        }
    }
    else{
        std::cout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return parameters;
}

template <int dim, int nstate>
void AdjointWeights<dim,nstate>::build_problem(){
    std::cout << "Solve for A and b for the NNLS Problem from POD Snapshots"<< std::endl;
    MatrixXd snapshotMatrix = this->pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = this->pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    Epetra_Map system_matrix_rowmap = epetra_system_matrix.RowMap();

    // Get dimensions of the problem
    int num_snaps_POD = snapshotMatrix.cols(); // Number of snapshots used to build the POD basis
    int n_reduced_dim_POD = epetra_pod_basis.NumGlobalCols(); // Reduced subspace dimension
    int num_elements_N_e = this->dg->triangulation->n_active_cells(); // Number of elements (equal to N if there is one DOF per cell)

    // Create empty and temporary C and d structs
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    int training_snaps;
    // Check if all or a subset of the snapshots will be used for training
    if (this->all_parameters->hyper_reduction_param.num_training_snaps != 0) {
        std::cout << "LIMITED NUMBER OF TRAINING SNAPSHOTS" << std::endl;
        training_snaps = this->all_parameters->hyper_reduction_param.num_training_snaps;
    }
    else{
        training_snaps = num_snaps_POD;
    }
    Epetra_Map RowMap((n_reduced_dim_POD*n_reduced_dim_POD*training_snaps), 0, epetra_comm); // Number of rows in Jacobian based training matrix = n^2 * (number of training snapshots)
    Epetra_Map ColMap(num_elements_N_e, 0, epetra_comm);

    int snap_num = 0;
    dealii::Vector<double> error_sum(this->dg->dof_handler.get_triangulation().n_active_cells());
    for(auto snap_param : this->snapshot_parameters.rowwise()){
        std::cout << "Snapshot Parameter Values" << std::endl;
        std::cout << snap_param << std::endl;
        dealii::LinearAlgebra::ReadWriteVector<double> snapshot_s;
        snapshot_s.reinit(num_elements_N_e);
        // Extract snapshot from the snapshotMatrix
        for (int snap_row = 0; snap_row < num_elements_N_e; snap_row++){
            snapshot_s(snap_row) = snapshotMatrix(snap_row, snap_num);
        }
        dealii::LinearAlgebra::distributed::Vector<double> reference_solution(this->dg->solution);
        reference_solution.import(snapshot_s, dealii::VectorOperation::values::insert);
        
        // Modifiy parameters for snapshot location and create new flow solver
        Parameters::AllParameters params = this->reinitParams(snap_param);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);
        this->dg = flow_solver->dg;

        // Set solution to snapshot and re-compute the residual/Jacobian
        this->dg->solution = reference_solution;
        const bool compute_dRdW = true;
        this->dg->assemble_residual(compute_dRdW);
        Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());

        dealii::Vector<double> estimated_error_per_cell(num_elements_N_e);
        std::unique_ptr<DualWeightedResidualError<dim, nstate, double>> mesh_error = std::make_unique<DualWeightedResidualError<dim, nstate, double>>(this->dg);
        estimated_error_per_cell = mesh_error->compute_cellwise_errors();
        
        std::cout << "Estimated Error per Cell" << std::endl;
        estimated_error_per_cell.print(std::cout);
        if (snap_num == 0){
            error_sum = estimated_error_per_cell;
        }
        else{
            error_sum.add(1.0,estimated_error_per_cell);
        }

        snap_num+=1;

        // Check if number of training snapshots has been reached
        if (this->all_parameters->hyper_reduction_param.num_training_snaps != 0) {
            std::cout << "LIMITED NUMBER OF SNAPSHOTS"<< std::endl;
            if (snap_num > (this->all_parameters->hyper_reduction_param.num_training_snaps-1)){
                break;
            }
        }
    }

    double percent_of_error = 0;
    dealii::Vector<double> error_weight(num_elements_N_e);
    double total_DWR_error = error_sum.l1_norm();

    std::vector<int> y(this->dg->dof_handler.get_triangulation().n_active_cells());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&error_sum](int a, int b){ return error_sum[a] > error_sum[b]; };
    std::sort(y.begin(), y.end(), comparator);

    for (unsigned int k = 0; k <  this->dg->dof_handler.get_triangulation().n_active_cells() ; k++){
        int j = y[k];
        double contrib = (error_sum[j]/ total_DWR_error) * 100;
        error_weight[j] = contrib;
        percent_of_error += contrib;
        std::cout << "percent_of_error" << std::endl;
        std::cout << percent_of_error << std::endl;
        if (percent_of_error > 99.0){
            break;
        }
    }
    weights = error_weight;
    std::cout << "Weights" << std::endl;
    weights.print(std::cout);
}

#if PHILIP_DIM==1
    template class AdjointWeights<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
    template class AdjointWeights<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // HyperReduction namespace
} // PHiLiP namespace