#include "headers.h"

extern double kernel_total_time;
extern int kernel_calls;
float max_deviation = 0.005;

void init_problem(cl_float3 *position_arr) {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
                if( count == particles_count){
                    return;
                }
                position_arr[count] = (cl_float3){ i, j, l };
                count++;
            }
        }
    }
    if( count < particles_count ){
        printf("error decrease initial_dist parameter, count is %ld  particles_count is %ld \n", count, particles_count);
        exit(1);
    }
}

void mc(cl_float3 *position_arr, float *energy_arr, cl_float3 *nearest) {
    int i = 0;
    int good_iter = 0;
    int good_iter_hung = 0;
    float energy_ar[nmax] = {};
    float u1 = calculate_energy_lj(position_arr, energy_arr, nearest);
    while (1) {
        if ((good_iter == nmax) || (i == total_it)) {
            printf("energy is %f \ngood iters percent %f \n", energy_ar[good_iter-1]/particles_count, (float)good_iter/(float)total_it);
            kernel_calls = i;
            break;
        }
        cl_float3 tmp[particles_count];
        memcpy(tmp, position_arr, sizeof(cl_float3)*particles_count);
        for (int particle = 0; particle < particles_count; particle++) {
            //offset between -max_deviation/2 and max_deviation/2
            double ex = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ey = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ez = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            position_arr[particle].x = position_arr[particle].x + ex;
            position_arr[particle].y = position_arr[particle].y + ex;
            position_arr[particle].z = position_arr[particle].z + ex;
        }
        double u2 = calculate_energy_lj(position_arr, energy_arr, nearest);
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

float calculate_energy_lj(cl_float3 *position_arr, float *energy_arr, cl_float3 *nearest) {
    nearest_image(position_arr, nearest);
    memset(energy_arr, 0, sizeof(energy_arr));
    run();
    float total_energy = 0;
    for (unsigned i = 0; i < particles_count; i++)
        total_energy+=energy_arr[i];
    total_energy/=2;
    return total_energy;
}

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