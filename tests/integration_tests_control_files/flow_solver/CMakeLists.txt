set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
#configure_file(inviscid_taylor_green_vortex.prm inviscid_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_INVISCID_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------

# =======================================
# Viscous Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex.prm viscous_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME SERIAL_VISCOUS_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex.prm viscous_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex_large_eddy_simulation.prm viscous_taylor_green_vortex_large_eddy_simulation.prm COPYONLY)
#add_test(
#  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_LARGE_EDDY_SIMULATION
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_large_eddy_simulation.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex_large_eddy_simulation.prm viscous_taylor_green_vortex_large_eddy_simulation.prm COPYONLY)
#add_test(
#  NAME SERIAL_VISCOUS_TAYLOR_GREEN_VORTEX_LARGE_EDDY_SIMULATION
#  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_large_eddy_simulation.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------

# =======================================
# SD7003 Airfoil
# =======================================
# ----------------------------------------
# - References:
# - (1) Uranga et al. 
# - (2) Garmann et al.
# ----------------------------------------
# configure_file(unsteady_sd7003.prm unsteady_sd7003.prm COPYONLY)
# add_test(
#  NAME MPI_3D_SD7003_UNSTEADY_FLOW
#  COMMAND mpirun -np ${MPI_MAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/unsteady_sd7003.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
# ----------------------------------------