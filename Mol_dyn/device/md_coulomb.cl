/**
 * @file md_coulomb.cl
 * @brief OpenCL kernel which calculate energy and force
 */

#include "parameters.h"
/**
 * @brief OpenCL kernel for coulomb potential
 * @param particles Position array
 * @param charge Charge array
 * @param out_energy Energy which describe how one particles iteract which all others
 * @param out_force Force acting on the particle from all others
 * @return void
 */
__attribute__((reqd_work_group_size(particles_count, 1, 1)))
__kernel void md(__global const float3 *restrict particles,
                 __global const int *restrict charge,
                 __global float *restrict out_energy,
                 __global float3 *restrict out_force) {

    int index = get_global_id(0);
    float energy = 0;
    float3 force = (float3)(0, 0, 0);
    #pragma unroll 2
    for (int i = 0; i < particles_count; i++) {
        float x = particles[i].x - particles[index].x;
        float y = particles[i].y - particles[index].y;
        float z = particles[i].z - particles[index].z;
        /* second part of implementation periodic boundary conditions */
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
        if (i != index) {
            float3 r = (float3)(x, y, z);
            float dist = fast_length(r);
            float inv_dist = native_divide(1, dist);
            float inv_dist_cub = native_divide(1, dist * dist * dist);
            if ((charge[index] == -1) || (charge[i] == -1)){
                float erf_arg = native_divide(dist, SIGMA);
                float multiplier = erf(erf_arg);
                float inv_dist_square = inv_dist * inv_dist;
                energy += charge[i] * charge[index] * native_divide(multiplier, dist);
                force += r * charge[i] * charge[index] * ((-DERIVATIVE_ERF * native_exp(-erf_arg * erf_arg) * inv_dist_square) + multiplier * inv_dist_cub);
            }
            else{
                energy += charge[i] * charge[index] * inv_dist;
                force += r * (charge[i] * charge[index] * inv_dist_cub);
            }
        }
    }
    out_force[index] = force;
    out_energy[index] = energy;
}