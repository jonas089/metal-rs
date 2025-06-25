use metal::*;
use std::mem;
use std::time::Instant;

use crate::constants::{CHUNK_SIZE, LIST_SIZE, NUM_CHUNKS, THREAD_GROUP_SIZE};
mod constants;

fn cpu_reduction(input: &[u64]) -> u64 {
    input.iter().copied().sum()
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("GPU: {}", device.name());
    println!("Max threads per threadgroup: {:?}", device.max_threads_per_threadgroup());
    
    let metallib_data = std::fs::read("build/sum.metallib").expect("missing .metallib file");
    let library = device.new_library_with_data(&metallib_data).unwrap();
    
    let sum_function = library.get_function("sum_reduction", None).unwrap();
    let sum_pipeline_descriptor = ComputePipelineDescriptor::new();
    sum_pipeline_descriptor.set_compute_function(Some(&sum_function));
    let sum_pipeline = device.new_compute_pipeline_state(&sum_pipeline_descriptor).unwrap();

    let max_threads_for_kernel = sum_pipeline.max_total_threads_per_threadgroup();
    
    println!("Max threads per group supported by kernel: {}", max_threads_for_kernel);
    let group_size = THREAD_GROUP_SIZE.min(max_threads_for_kernel);
    let input: Vec<u64> = (0..LIST_SIZE).map(|i| (i + 1) as u64).collect();
    
    println!("\n=== CPU Computation ===");
    let cpu_start = Instant::now();
    println!("Starting CPU computation at: {:?}", cpu_start);
    let cpu_result = cpu_reduction(&input);
    let cpu_end = Instant::now();
    let cpu_duration = cpu_end.duration_since(cpu_start);
    println!("CPU computation completed at: {:?}", cpu_end);
    println!("CPU execution time: {:?}", cpu_duration);
    println!("CPU result: {}", cpu_result);

    println!("\n=== GPU Computation ===");
    let gpu_start = Instant::now();
    println!("Starting GPU computation at: {:?}", gpu_start);
    let mut partial_sums: Vec<u64> = Vec::with_capacity(NUM_CHUNKS);
    let command_queue = device.new_command_queue();
    for chunk_idx in 0..NUM_CHUNKS {
        let chunk_start = chunk_idx * CHUNK_SIZE;
        let chunk_end = (chunk_idx + 1) * CHUNK_SIZE;
        let chunk_data = &input[chunk_start..chunk_end];
        let chunk_buf = device.new_buffer_with_data(
            chunk_data.as_ptr() as *const _,
            CHUNK_SIZE as u64 * mem::size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let max_intermediate_size = (CHUNK_SIZE + group_size as usize - 1) / group_size as usize;
        let intermediate_buf = device.new_buffer(
            max_intermediate_size as u64 * mem::size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut num_elements = CHUNK_SIZE as u64;
        let mut input_buf = chunk_buf;
        let mut output_buf = intermediate_buf;
        
        while num_elements > 1 {
            let group_size = THREAD_GROUP_SIZE.min(max_threads_for_kernel);
            println!("Group size: {}", group_size);
            let num_groups = (num_elements + group_size as u64 - 1) / group_size as u64;
            unsafe {
                let output_slice = std::slice::from_raw_parts_mut(
                    output_buf.contents() as *mut u64,
                    num_groups as usize
                );
                for val in output_slice.iter_mut() {
                    *val = 0;
                }
            }
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&sum_pipeline);
            encoder.set_buffer(0, Some(&input_buf), 0);
            encoder.set_buffer(1, Some(&output_buf), 0);
            
            let numel_buf = device.new_buffer_with_data(
                &num_elements as *const u64 as *const _,
                std::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(2, Some(&numel_buf), 0);
            let numgroups_buf = device.new_buffer_with_data(
                &num_groups as *const u64 as *const _,
                std::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(3, Some(&numgroups_buf), 0);
            encoder.set_threadgroup_memory_length(0, group_size as u64 * mem::size_of::<u64>() as u64);
            let grid = MTLSize {
                width: num_groups * group_size as u64,
                height: 1,
                depth: 1,
            };
            let threads_per_group = MTLSize {
                width: group_size as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, threads_per_group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            num_elements = num_groups;
            std::mem::swap(&mut input_buf, &mut output_buf);
        }
        let chunk_result = unsafe {
            *(input_buf.contents() as *const u64)
        };
        partial_sums.push(chunk_result);
    }
    let gpu_result = partial_sums.iter().sum::<u64>();
    let gpu_end = Instant::now();
    let gpu_duration = gpu_end.duration_since(gpu_start);
    println!("GPU computation completed at: {:?}", gpu_end);
    println!("GPU execution time: {:?}", gpu_duration);
    println!("GPU result: {}", gpu_result);

    println!("\n=== Performance Comparison ===");
    println!("CPU time: {:?}", cpu_duration);
    println!("GPU time: {:?}", gpu_duration);
    if gpu_duration < cpu_duration {
        let speedup = cpu_duration.as_nanos() as f64 / gpu_duration.as_nanos() as f64;
        println!("GPU is {:.2}x faster than CPU", speedup);
    } else {
        let slowdown = gpu_duration.as_nanos() as f64 / cpu_duration.as_nanos() as f64;
        println!("CPU is {:.2}x faster than GPU", slowdown);
    }

    println!("\n=== Result Verification ===");
    let difference = if cpu_result > gpu_result { cpu_result - gpu_result } else { gpu_result - cpu_result };
    assert_eq!(difference, 0);
}
