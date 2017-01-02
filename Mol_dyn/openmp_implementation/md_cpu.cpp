#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <sys/timeb.h>
#include <omp.h>
#include <string.h>
#include "parameters.h"
#include "logger.h"

#define NUM_THREADS 8
#define COULOMB "--coulomb"

struct dim {
    double x;
    double y;
    double z;
};
typedef struct dim dim;

void nearest_image(dim *position_arr, dim *nearest);
void init_problem(dim *position_arr, dim *velocity, dim *output_force, int *charge);
void md(dim *position_arr, dim *velocity, dim *output_force, dim *nearest, int *charge);
double calculate_energy_force_lj(dim *position_arr, dim *nearest, dim *output_force, int *charge);
double calculate_energy_force_coulomb(dim *position_arr, dim *nearest, dim *output_force, int *charge);
void motion(dim *position_arr, dim *velocity, dim *output_force);

double (*calculate_energy_force)(dim*, dim*, dim*, int*);

int main(int argc, char *argv[])
{
    calculate_energy_force = calculate_energy_force_lj;
    if (argc > 1){
        if (!strcmp(argv[1], COULOMB)){
            calculate_energy_force = calculate_energy_force_coulomb;
        }
    }
    struct timeb start_total_time;
    ftime(&start_total_time);
    dim *position_arr = (dim*)malloc(sizeof(dim) * particles_count);
    dim *nearest = (dim*)malloc(sizeof(dim) * particles_count);
    dim *velocity = (dim*)malloc(sizeof(dim) * particles_count);
    dim *output_force = (dim*)malloc(sizeof(dim) * particles_count);
    int *charge = (int*)malloc(sizeof(int) * particles_count);

    init_problem(position_arr, velocity,output_force, charge);
    md(position_arr, velocity, output_force, nearest, charge);

    free(position_arr);
    free(nearest);
    free(velocity);
    free(output_force);
    free(charge);
    struct timeb end_total_time;
    ftime(&end_total_time);
    printf("Total execution time in ms =  %d", (int)((end_total_time.time - start_total_time.time) * 1000 + end_total_time.millitm - start_total_time.millitm));
    LOG_PRINT("Total execution time in ms =  %d", (int)((end_total_time.time - start_total_time.time) * 1000 + end_total_time.millitm - start_total_time.millitm));
    return 0;
}

/////// HELPER FUNCTIONS ///////

void init_problem(dim *position_arr, dim *velocity, dim *output_force, int *charge) {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
                if( count == particles_count){
                    return;
                }
                position_arr[count] = { i,j,l };
                velocity[count] = { 0, 0, 0 };
                output_force[count] = { 0, 0, 0 };
                if (calculate_energy_force == calculate_energy_force_coulomb){
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

void nearest_image(dim *position_arr, dim *nearest){
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
        nearest[i] = (dim){ x, y, z};
    }
}

double calculate_energy_force_lj(dim *position_arr, dim *nearest, dim *output_force, int *charge){
    for (int i = 0; i < particles_count; i++){
        output_force[i] = { 0, 0, 0};
    }
    nearest_image(position_arr, nearest);
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(NUM_THREADS)
    for (int i = 0; i < particles_count; i++) {
        double force_x = 0;
        double force_y = 0;
        double force_z = 0;
        for (int j = 0; j < particles_count; j++) {
            float x = nearest[j].x - nearest[i].x;
            float y = nearest[j].y - nearest[i].y;
            float z = nearest[j].z - nearest[i].z;
            if (x > half_box)
                x -= box_size;
            else {
                if (x < -half_box)
                    x += box_size;
            }
            if (y > half_box)
                y -= box_size;
            else {
                if (y < -half_box)
                    y += box_size;
            }
            if (z > half_box)
                z -= box_size;
            else {
                if (z < -half_box)
                    z += box_size;
            }
            float sq_dist = x * x + y * y + z * z;
            if ((sq_dist < rc * rc) && (i != j)) {
                double r6 = sq_dist * sq_dist * sq_dist;
                double r12 = r6 * r6;
                double r8 = r6 * sq_dist;
                double r14 = r12 * sq_dist;
                double multiplier = (24 * (2 / r14 - 1 / r8));
                force_x += x * multiplier;
                force_y += y * multiplier;
                force_z += z * multiplier;
                energy += 4 * (1 / r12 - 1 / r6);
            }
        }
        output_force[i].x = force_x;
        output_force[i].y = force_y;
        output_force[i].z = force_z;
    }
    return energy / 2;
}

double calculate_energy_force_coulomb(dim *position_arr, dim *nearest, dim *output_force, int *charge){
    for (int i = 0; i < particles_count; i++){
        output_force[i] = { 0, 0, 0};
    }
    nearest_image(position_arr, nearest);
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(NUM_THREADS)
    for (int i = 0; i < particles_count; i++) {
        double force_x = 0;
        double force_y = 0;
        double force_z = 0;
        for (int j = 0; j < particles_count; j++) {
            float x = nearest[j].x - nearest[i].x;
            float y = nearest[j].y - nearest[i].y;
            float z = nearest[j].z - nearest[i].z;
            if (x > half_box)
                x -= box_size;
            else {
                if (x < -half_box)
                    x += box_size;
            }
            if (y > half_box)
                y -= box_size;
            else {
                if (y < -half_box)
                    y += box_size;
            }
            if (z > half_box)
                z -= box_size;
            else {
                if (z < -half_box)
                    z += box_size;
            }
            if (i != j ) {
                double sq_dist = x * x + y * y + z * z;
                double dist = sqrt(sq_dist);
                double multiplier = (charge[i] * charge[j] / sq_dist) / dist;
                force_x += x * multiplier;
                force_y += y * multiplier;
                force_z += z * multiplier;
                energy += charge[i] * charge[j] / sqrt(sq_dist);
            }
        }
        output_force[i].x = force_x;
        output_force[i].y = force_y;
        output_force[i].z = force_z;
    }
    return energy / 2;
}

void md(dim *position_arr, dim *velocity, dim *output_force, dim *nearest, int *charge) {
    for (int n = 0; n < total_it; n ++){
        double total_energy = calculate_energy_force(position_arr, nearest, output_force, charge);
        motion(position_arr, velocity, output_force);
        if (n == (total_it - 1)) {
            printf("energy is %f \n", total_energy/particles_count);
            LOG_PRINT("energy is %f", total_energy/particles_count);
        }
    }
}

void motion(dim *position_arr, dim *velocity, dim * output_force){
    for (int i = 0; i < particles_count; i++) {
        velocity[i] = {velocity[i].x + output_force[i].x * dt,
            velocity[i].y + output_force[i].y * dt,
            velocity[i].z + output_force[i].z * dt};
        position_arr[i] = {position_arr[i].x + velocity[i].x * dt,
            position_arr[i].y + velocity[i].y * dt,
            position_arr[i].z + velocity[i].z * dt};
    }
}
