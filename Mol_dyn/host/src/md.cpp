/**
 * @file md.cpp
 * @brief Implement MD algorithm
 */

/*
 * Includes
 */
#include "headers.h"

extern void (*run)();
extern cl_float final_energy;

/**
 * @brief set initial coordinates,velocities and charges for all particles
 * @param position_arr Position array
 * @param velocity Velocity array
 * @param charge Charge array
 * @return void
 */
void init_problem(cl_float3 *position_arr, cl_float3 *velocity, cl_int *charge) {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
                if( count == particles_count){
                    return;
                }
                position_arr[count] = (cl_float3){ i, j, l };
                velocity[count] = (cl_float3){ 0, 0, 0 };
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
 * @brief solve motion equation's using Euler method
 * @param position_arr Position array
 * @param velocity Velocity array
 * @param output_force force array
 * @return void
 */
void motion(cl_float3 *position_arr, cl_float3 *velocity, cl_float3 *output_force) {
    for (int i = 0; i < particles_count; i++) {
        /* v+= f * dt */
        velocity[i] = (cl_float3) {velocity[i].x + output_force[i].x * dt,
            velocity[i].y + output_force[i].y * dt,
            velocity[i].z + output_force[i].z * dt};
        /* r+= v * dt */
        position_arr[i] = (cl_float3) {position_arr[i].x + velocity[i].x * dt,
            position_arr[i].y + velocity[i].y * dt,
            position_arr[i].z + velocity[i].z * dt};
    }
}

/**
 * @brief calculate energy and force on device
 * @param position_arr Position array
 * @param nearest nearest array
 * @param output_force force array, calculated on device
 * @param output_energy energy array
 * @param charge array Charge array
 * @return void
 */
void calculate_energy_force(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_int *charge) {
    nearest_image(position_arr, nearest);
    for (int i = 0; i < particles_count; i++){
        output_force[i] = (cl_float3){0, 0, 0};
        output_energy[i] = 0;
    }
    /** run kernel */
    run();
}

/**
 * @brief perform MD iterations
 * @param position_arr Position array
 * @param nearest nearest array
 * @param output_force force array, calculated on device
 * @param output_energy energy array
 * @param charge array Charge array
 * @return void
 */
void md(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_float3 *velocity, cl_int *charge) {
    for (int n = 0; n < total_it; n ++){
        calculate_energy_force(position_arr, nearest, output_force, output_energy, charge);
        motion(position_arr, velocity, output_force);
        float total_energy = 0;
        for (int i = 0; i < particles_count; i++)
            total_energy+=output_energy[i];
        total_energy/=(2 * particles_count);
        if (n == (total_it - 1)){
            final_energy = total_energy;
        }
    }
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
