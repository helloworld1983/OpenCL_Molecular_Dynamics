/**
 * @file md_lj.cl
 * @brief OpenCL kernel which calculate energy and force
 */

#include "parameters.h"
/**
 * @brief OpenCL kernel for LJ
 * @param particles Position array
 * @param out_energy Energy which describe how one particles iteract which all others
 * @param out_force Force acting on the particle from all others
 * @return void
 */
__attribute__((reqd_work_group_size(particles_count, 1, 1)))
__kernel void md(__global const float3 *restrict particles,
                 __global float *restrict out_energy,
                 __global float3 *restrict out_force) {

    int index = get_global_id(0);
    float energy = 0;
    float3 force = (float3)(0, 0, 0);
    #pragma unroll 4
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
        float3 r = (float3)(x, y, z);
        float sq_dist = x * x + y * y + z * z;
        if ((sq_dist < (rc * rc)) && (i != index)) {
            float r6 = sq_dist * sq_dist * sq_dist;
            float r12 = r6 * r6;
            float r8 = r6 * sq_dist;
            float r14 = r12 * sq_dist;
            force += r * (24 * (2 / r14 - 1 / r8));
            energy += 4 * (1 / r12 - 1 / r6);
        }
    }
    out_force[index] = force;
    out_energy[index] = energy;
}