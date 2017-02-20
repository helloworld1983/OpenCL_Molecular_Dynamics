/**
 * @file mc_coulomb.cl
 * @brief OpenCL kernel which calculate energy
 */

#include "parameters.h"
/**
 * @brief OpenCL kernel for coulomb potential
 * @param particles Position array
 * @param charge Charge array
 * @param out_energy Energy which describe how one particles iteract which all others
 * @return void
 */
__attribute__((reqd_work_group_size(particles_count, 1, 1)))
__kernel void mc(__global const float3 *restrict particles,
                 __global const int *restrict charge,
                 __global float *restrict out_energy) {
    int index = get_global_id(0);
    float energy = 0;
    #pragma unroll 4
    for (int i = 0; i < particles_count; i++) {
        float x = particles[i].x - particles[index].x;
        float y = particles[i].y - particles[index].y;
        float z = particles[i].z - particles[index].z;
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
            if ((charge[index] == -1) || (charge[i] == -1)){
                float erf_arg = native_divide(dist, SIGMA);
                float multiplier = erf(erf_arg);
                energy += charge[i] * charge[index] * multiplier * inv_dist;
            }
            else{
                energy += charge[i] * charge[index] * inv_dist;
            }
        }
    }
    out_energy[index] = energy;
}

