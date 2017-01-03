#include "parameters.h"

__attribute__((reqd_work_group_size(particles_count, 1, 1)))
__kernel void md(__global const float3 *restrict particles,
                 __global const int *restrict charge,
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
            energy += charge[i] * charge[index] / dist;
            float dist_cub = dist * dist * dist;
            force += r * (charge[i] * charge[index] / dist_cub);
        }
    }
    out_force[index] = force;
    out_energy[index] = energy;
}