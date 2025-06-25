#include <metal_stdlib>
using namespace metal;

kernel void sum_reduction(
    device const ulong* input  [[buffer(0)]],
    device ulong* output       [[buffer(1)]],
    device const uint* n       [[buffer(2)]],
    device const uint* num_groups [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]],
    uint lid                  [[thread_position_in_threadgroup]],
    uint group_id             [[threadgroup_position_in_grid]],
    uint tg_size              [[threads_per_threadgroup]],
    threadgroup ulong* localSums [[threadgroup(0)]]
) {
    uint numel = n[0];
    uint num_output_groups = num_groups[0];

    // Load input or zero if out-of-bounds
    ulong val = (gid < numel) ? input[gid] : 0;
    localSums[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = tg_size;

    while (active > 1) {
        if (lid < active / 2) {
            localSums[lid] += localSums[lid + (active + 1) / 2]; // handle odd lengths
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = (active + 1) / 2;
    }

    if (lid == 0 && group_id < num_output_groups) {
        output[group_id] = localSums[0];
    }
}
