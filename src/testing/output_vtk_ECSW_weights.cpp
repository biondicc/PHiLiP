#include "output_vtk_ECSW_weights.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "reduced_order/hyper_reduced_adaptive_sampling.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {


template <int dim, int nstate>
OutputVTKWeights<dim, nstate>::OutputVTKWeights(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
bool OutputVTKWeights<dim, nstate>::getWeightsFromFile() const{
    bool file_found = false;
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    VectorXd weights_eig;
    int rows = 0;
    std::string path = all_parameters->reduced_order_param.path_to_search; 

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("weights") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;

            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else {
                        try{
                            std::stod(field);
                            rows++;
                        } catch (...){
                            continue;
                        }
                    }
                }
            }

            weights_eig.resize(rows);
            int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else {
                        try{
                            double num_string = std::stod(field);
                            std::cout << field << std::endl;
                            weights_eig(row) = num_string;
                            row++;
                        } catch (...){
                            continue;
                        }
                    }
                }
            }
            myfile.close();
        }
    }

    Epetra_Map RowMap(rows, 0, epetra_comm);
    Epetra_Vector weights(RowMap);
    for(int i = 0; i < rows; i++){
        weights[i] = weights_eig(i);
    }

    ptr_weights = std::make_shared<Epetra_Vector>(weights);
    return file_found;
}

template <int dim, int nstate>
int OutputVTKWeights<dim, nstate>::run_test() const
{
    pcout << "Load ECSW Weights and output VTK File..." << std::endl;

    // Create POD Petrov-Galerkin ROM with Hyper-reduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);

    std::shared_ptr<HyperreducedAdaptiveSampling<dim,nstate>> hyper_reduced_ROM_solver = std::make_unique<HyperreducedAdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);

    bool weights_found = getWeightsFromFile();
    if (weights_found){
        std::cout << "ECSW Weights" << std::endl;
        std::cout << *ptr_weights << std::endl;
    }
    else{
        std::cout << "File with weights not found in folder" << std::endl;
        return -1;
    }

    // hyper_reduced_ROM_solver->placeROMLocations(rom_points, *ptr_weights);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    
    int num_elements_N_e = flow_solver->dg->triangulation->n_active_cells();
    dealii::Vector<double> weights_dealii(num_elements_N_e);
    Epetra_Vector epetra_weights = *ptr_weights;
    for(int j = 0 ; j < epetra_weights.GlobalLength() ; j++){
        weights_dealii[j] = epetra_weights[j];
    } 
    
    flow_solver->dg->reduced_mesh_weights = weights_dealii;
    flow_solver->dg->output_results_vtk(1000);
    hyper_reduced_ROM_solver->outputIterationData("HROM_post_sampling");
    return 0;
}

#if PHILIP_DIM==1
        template class OutputVTKWeights<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class OutputVTKWeights<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
