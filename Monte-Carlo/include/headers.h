#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include "parameters.h"
#include <string.h>

#define MAX_PLATFORMS_COUNT 2

void init_problem(cl_float3 *input);
bool init_opencl();
void run();
void cleanup();
void mc(cl_float3 *position_arr, float *energy_arr, cl_float3 *nearest);
void nearest_image(cl_float3 *position_arr, cl_float3 *nearest);
float calculate_energy_lj(cl_float3 *position_arr, float *energy_arr, cl_float3 *nearest);