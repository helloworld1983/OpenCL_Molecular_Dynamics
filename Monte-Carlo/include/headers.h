/**
 * @file headers.h
 * @brief includes and functions prototypes
 */

/*
 * Includes
 */
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>
#include "parameters.h"
#include <string.h>

#define MAX_PLATFORMS_COUNT 2

/**
 * Prototypes
 */
bool init_opencl_lj();
bool init_opencl_coulomb();
void run_lj();
void run_coulomb();
void cleanup();
void init_problem(cl_float3 *input, cl_int *charge);
void mc(cl_float3 *position_arr, cl_float *energy_arr, cl_float3 *nearest, cl_int *charge);
void nearest_image(cl_float3 *position_arr, cl_float3 *nearest);
cl_float calculate_energy(cl_float3 *position_arr, cl_float *energy_arr, cl_float3 *nearest, cl_int *charge);