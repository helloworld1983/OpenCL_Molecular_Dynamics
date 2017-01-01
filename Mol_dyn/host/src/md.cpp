#include "headers.h"

extern void (*run)();

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

void motion(cl_float3 *position_arr, cl_float3 *velocity, cl_float3 *output_force) {
    for (int i = 0; i < particles_count; i++) {
        velocity[i] = (cl_float3) {velocity[i].x + output_force[i].x * dt,
            velocity[i].y + output_force[i].y * dt,
            velocity[i].z + output_force[i].z * dt};
        position_arr[i] = (cl_float3) {position_arr[i].x + velocity[i].x * dt,
            position_arr[i].y + velocity[i].y * dt,
            position_arr[i].z + velocity[i].z * dt};
    }
}

void calculate_energy_force(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_int *charge) {
    nearest_image(position_arr, nearest);
    for (int i = 0; i < particles_count; i++){
        output_force[i] = (cl_float3){0, 0, 0};
        output_energy[i] = 0;
    }
    run();
}

void md(cl_float3 *position_arr, cl_float3 *nearest, cl_float3 *output_force, cl_float *output_energy, cl_float3 *velocity, cl_int *charge) {
    for (int n = 0; n < total_it; n ++){
        calculate_energy_force(position_arr, nearest, output_force, output_energy, charge);
        motion(position_arr, velocity, output_force);
        float total_energy = 0;
        for (int i = 0; i < particles_count; i++)
            total_energy+=output_energy[i];
        total_energy/=(2 * particles_count);
        if (n == (total_it - 1)){
            printf("energy is %f \n",total_energy);
        }
    }
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
