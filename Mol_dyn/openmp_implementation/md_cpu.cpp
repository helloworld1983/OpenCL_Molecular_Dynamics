#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <sys/timeb.h>
#include <omp.h>
#include <string.h>
#include "parameters.h"

#define NUM_THREADS 8

struct dim {
    double x;
    double y;
    double z;
};
typedef struct dim dim;

void nearest_image(dim *initial_array, dim *nearest);
void set_initial_state(dim *array, dim *velocity, dim *force);
double fast_pow(double a, int n);
void md(dim *array, dim *velocity, dim *force);
double calculate_energy_force_lj(dim *array, dim *force);
void motion(dim *array, dim *velocity, dim *force);

double Urc = 4 * ( 1 / fast_pow(rc, 12) - 1 / fast_pow(rc, 6) );

int main()
{
    struct timeb start_total_time;
    ftime(&start_total_time);
    dim *r = (dim*)malloc(sizeof(dim) * size);
    dim *v = (dim*)malloc(sizeof(dim) * size);
    dim *f = (dim*)malloc(sizeof(dim) * size);
    set_initial_state(r,v,f);
    md(r,v,f);
    free(r);
    free(v);
    free(f);
    struct timeb end_total_time;
    ftime(&end_total_time);
    printf("\nTotal execution time in ms =  %d\n", (int)((end_total_time.time - start_total_time.time) * 1000 + end_total_time.millitm - start_total_time.millitm));
    return 0;
}

/////// HELPER FUNCTIONS ///////

void set_initial_state(dim *array, dim *velocity, dim *force) {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
                if( count == size){
                    return; //it is not balanced grid but we can use it
                }
                array[count] = { i,j,l };
                velocity[count] = { 0, 0, 0 };
                force[count] = { 0, 0, 0 };
                count++;
            }
        }
    }
    if( count < size ){
        printf("error decrease initial_dist parameter, count is %ld  size is %ld \n", count, size);
        exit(1);
    }
}

void nearest_image(dim *array, dim *nearest){
    for (int i = 0; i < size; i++){
        float x,y,z;
        if (array[i].x  > 0){
            x = fmod(array[i].x + half_box, box_size) - half_box;
        }
        else{
            x = fmod(array[i].x - half_box, box_size) + half_box;
        }
        if (array[i].y  > 0){
            y = fmod(array[i].y + half_box, box_size) - half_box;
        }
        else{
            y = fmod(array[i].y - half_box, box_size) + half_box;
        }
        if (array[i].z  > 0){
            z = fmod(array[i].z + half_box, box_size) - half_box;
        }
        else{
            z = fmod(array[i].z - half_box, box_size) + half_box;
        }
        nearest[i] = (dim){ x, y, z};
    }
}


double calculate_energy_force_lj(dim *array, dim *force){
    for (int i = 0; i < size; i++){
        force[i] = { 0, 0, 0};
    }
    dim *nearest = (dim*)malloc(sizeof(dim) * size);
    nearest_image(array, nearest);
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(NUM_THREADS)
    for (int i = 0; i < size; i++) {
        double force_x = 0;
        double force_y = 0;
        double force_z = 0;
        for (int j = 0; j < size; j++) {
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
            if ((sq_dist < rc * rc) && (j != i)) {
                double r6 = sq_dist * sq_dist * sq_dist;
                double r12 = r6 * r6;
                double r8 = r6 * sq_dist;
                double r14 = r12 * sq_dist;
                double multiplier = (12 * (1 / r14 - 1 / r8));
                force_x += x * multiplier;
                force_y += y * multiplier;
                force_z += z * multiplier;
                energy += 4 * (1 / r12 - 1 / r6) - Urc;
            }
        }
        force[i].x = force_x;
        force[i].y = force_y;
        force[i].z = force_z;
    }
    free(nearest);
    // now we consider each interaction twice, so we need to divide energy by 2
    return energy / 2;
}

void md(dim *array, dim *velocity, dim *force) {
    for (int n = 0; n < total_it; n ++){
        double total_energy = calculate_energy_force_lj(array, force);
        motion(array, velocity, force);
        if (!(n % 1000)) {
            printf("energy is %f \n", total_energy/size);
        }
    }
}

void motion(dim *array, dim *velocity, dim * force){
    for (int i = 0; i < size; i++) {
        velocity[i] = {velocity[i].x + force[i].x * dt,
            velocity[i].y + force[i].y * dt,
            velocity[i].z + force[i].z * dt};
        array[i] = {array[i].x + velocity[i].x * dt,
            array[i].y + velocity[i].y * dt,
            array[i].z + velocity[i].z * dt};
    }
}

inline double fast_pow(double a, int n) {
    if (n == 0)
        return 1;
    if (n % 2 == 1)
        return fast_pow(a, n - 1) * a;
    else {
        double b = fast_pow(a, n / 2);
        return b * b;
    }
}
