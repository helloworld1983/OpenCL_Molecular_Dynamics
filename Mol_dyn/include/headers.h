#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include "parameters.h"
#include <string.h>

#define MAX_PLATFORMS_COUNT 2
#define COULOMB "--coulomb"

bool init_opencl_lj();
bool init_opencl_coulomb();
void run_lj();
void run_coulomb();
void cleanup();
void init_problem(cl_float3 *position_arr, cl_float3 *velocity, cl_int *charge);
void md(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_float3 *velocity, cl_int *charge);
void nearest_image(cl_float3 *position_arr, cl_float3 *nearest);
void calculate_energy_force(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_int *charge);
void motion(cl_float3 *position_arr, cl_float3 *velocity, cl_float3 *output_force);