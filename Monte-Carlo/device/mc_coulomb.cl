#include "parameters.h"

__attribute__((reqd_work_group_size(particles_count, 1, 1)))
__kernel void mc(__global const float3 *restrict particles,
                 __global const int *restrict charge,
                 __global float *restrict out) {
    int index = get_global_id(0);
    float energy = 0;
    #pragma unroll 8
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
            energy += charge[i] * charge[index] / fast_length(r);
        }
    }
    out[index] = energy;
}

