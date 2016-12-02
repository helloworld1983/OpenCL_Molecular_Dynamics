#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include "parameters.h"
#include <string.h>

#define MAX_PLATFORMS_COUNT 2

bool init_opencl();
void run();
void cleanup();
void init_problem(cl_float3 *position_arr, cl_float3 *velocity);
void md(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, float *output_energy, cl_float3 *velocity);
void nearest_image(cl_float3 *position_arr, cl_float3 *nearest);
float calculate_energy_lj(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, float *output_energy);
void motion(cl_float3 *position_arr, cl_float3 *velocity, cl_float3 *output_force);