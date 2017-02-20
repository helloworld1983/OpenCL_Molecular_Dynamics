/**
 * @file md.cpp
 * @brief Implement MD algorithm
 */

/*
 * Includes
 */
#include "headers.h"
/** needs to print information in main.cpp */
extern double kernel_total_time;
extern int kernel_calls;
extern cl_float final_energy;
extern float good_iters_percent;
extern void (*run)();
float max_deviation = 0.007;

/**
 * @brief set initial coordinates and charges for all particles
 * @param position_arr Position array
 * @param charge Charge array
 * @return void
 */
void init_problem(cl_float3 *position_arr, cl_int *charge) {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
                if( count == particles_count ){
                    return;
                }
                position_arr[count] = (cl_float3){ i, j, l };
                if (run == run_coulomb){
                    if (count & 1)
                        charge[count] = 1;
                    else
                        charge[count] = -1;
                }
                count++;
            }
        }
    }
    if( count < particles_count ){
        printf("error decrease initial_dist parameter, count is %ld  particles_count is %ld \n", count, particles_count);
        exit(1);
    }
}

/**
 * @brief perform MC iterations
 * @param position_arr Position array
 * @param energy_arr energy array
 * @param nearest nearest array
 * @param charge array Charge array
 * @return void
 */
void mc(cl_float3 *position_arr, cl_float *energy_arr, cl_float3 *nearest, cl_int *charge) {
    int i = 0;
    int good_iter = 0;
    int good_iter_hung = 0;
    float energy_ar[nmax] = {};
    float u1 = calculate_energy(position_arr, energy_arr, nearest, charge);
    while (1) {
        if ((good_iter == nmax) || (i == total_it)) {
            final_energy = energy_ar[good_iter-1]/particles_count;
            good_iters_percent = (float)good_iter/(float)total_it;
            kernel_calls = i;
            break;
        }
        cl_float3 tmp[particles_count];
        memcpy(tmp, position_arr, sizeof(cl_float3)*particles_count);
        for (int particle = 0; particle < particles_count; particle++) {
            /** offset between -max_deviation/2 and max_deviation/2 */
            double ex = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ey = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ez = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            position_arr[particle].x = position_arr[particle].x + ex;
            position_arr[particle].y = position_arr[particle].y + ex;
            position_arr[particle].z = position_arr[particle].z + ex;
        }
        double u2 = calculate_energy(position_arr, energy_arr, nearest, charge);
        double deltaU_div_T = (u1 - u2) / Temperature;
        double probability = exp(deltaU_div_T);
        double rand_0_1 = (double)rand() / (double)RAND_MAX;
        if ((u2 < u1) || (probability <= rand_0_1)) {
            u1 = u2;
            energy_ar[good_iter] = u2;
            good_iter++;
            good_iter_hung++;
        }
        else {
            memcpy(position_arr, tmp, sizeof(cl_float3) * particles_count);
        }
        i++;
    }
}

/**
 * @brief calculate energy on device
 * @param position_arr Position array
 * @param energy_arr energy array
 * @param nearest nearest array
 * @param charge array Charge array
 * @return void
 */
cl_float calculate_energy(cl_float3 *position_arr, cl_float *energy_arr, cl_float3 *nearest, cl_int *charge) {
    nearest_image(position_arr, nearest);
    memset(energy_arr, 0, sizeof(energy_arr));
    run();
    float total_energy = 0;
    for (unsigned i = 0; i < particles_count; i++)
        total_energy+=energy_arr[i];
    total_energy/=2;
    return total_energy;
}

/**
 * @brief first part of implementation of periodic boundary conditions
 * @param position_arr Position array
 * @param nearest nearest array
 * @return void
 */
void nearest_image(cl_float3 *position_arr, cl_float3 *nearest){
    for (int i = 0; i < particles_count; i++){
        float x,y,z;
        if (position_arr[i].x  > 0){
            x = fmod(position_arr[i].x + half_box, box_size) - half_box;
        }
        else{
            x = fmod(position_arr[i].x - half_box, box_size) + half_box;
        }
        if (position_arr[i].y  > 0){
            y = fmod(position_arr[i].y + half_box, box_size) - half_box;
        }
        else{
            y = fmod(position_arr[i].y - half_box, box_size) + half_box;
        }
        if (position_arr[i].z  > 0){
            z = fmod(position_arr[i].z + half_box, box_size) - half_box;
        }
        else{
            z = fmod(position_arr[i].z - half_box, box_size) + half_box;
        }
        nearest[i] = (cl_float3){ x, y, z};
    }
}