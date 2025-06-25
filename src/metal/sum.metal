#include <metal_stdlib>
using namespace metal;

kernel void sum_reduction(
    device const ulong* input  [[buffer(0)]],
    device ulong* output       [[buffer(1)]],
    device const uint* n       [[buffer(2)]], // number of input elements
    device const uint* num_groups [[buffer(3)]], // number of output elements
    uint gid                  [[thread_position_in_grid]],
    uint lid                  [[thread_position_in_threadgroup]],
    uint group_id             [[threadgroup_position_in_grid]],
    uint tg_size              [[threads_per_threadgroup]],
    threadgroup ulong* localSums [[threadgroup(0)]]
) {
    uint numel = n[0];
    uint num_output_groups = num_groups[0];    
    ulong local_sum = 0;
    if (gid < numel) {
        local_sum = input[gid];
    }
    localSums[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            localSums[lid] += localSums[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0 && group_id < num_output_groups) {
        output[group_id] = localSums[0];
    }
}