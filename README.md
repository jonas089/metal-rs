# Metal-RS: GPU-Accelerated Array Reduction

A Rust project demonstrating GPU acceleration using Apple's Metal framework for parallel array reduction operations.

>[!NOTE]
> This project is just an experiment and requires a Macbook with an Apple silicon (M-series) chip.
> You also need to install Xcode before you can experiment with this code

## 🚀 Features

- **GPU Acceleration**: Uses Apple's Metal framework for parallel computation
- **Performance Comparison**: Benchmarks CPU vs GPU execution times
- **Large Scale Processing**: Handles arrays with 1 billion elements
- **Chunked Processing**: Processes data in manageable chunks to optimize memory usage
- **Cross-Platform**: Works on macOS with Metal support

## 📋 Prerequisites

- **macOS**: This project requires macOS with Metal support
- **Xcode Command Line Tools**: For Metal shader compilation
- **Rust**: Latest stable version (2024 edition)

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jonas089/metal-rs
   cd metal-rs
   ```

2. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

4. **Create build directory**:
   ```bash
   mkdir build
   ```

## 🏃‍♂️ Running the Project

1. **Compile the Metal shader**:
   ```bash
   make compile
   ```
   This compiles `src/metal/sum.metal` to `build/sum.metallib`

2. **Run the CPU vs GPU benchmark**:
   ```bash
   make run
   ```
   Or directly with Cargo:
   ```bash
   cargo run --release
   ```

## 📊 What It Does

This project demonstrates a parallel reduction algorithm that sums a large array of numbers:

- **Input**: Array of 1 billion sequential numbers (1, 2, 3, ..., 1,000,000,000)
- **Processing**: Data is processed in chunks of 100 million elements
- **Algorithm**: Uses a parallel reduction pattern with threadgroup memory
- **Output**: Total sum and performance comparison between CPU and GPU

### GPU Implementation Details

- **Metal Shader**: `sum_reduction` kernel performs parallel reduction
- **Thread Groups**: Uses 32 threads per threadgroup (configurable)
- **Memory**: Leverages threadgroup memory for efficient local reductions
- **Chunking**: Processes data in chunks to manage memory efficiently

## 📈 Performance

The application will output:
- CPU execution time and result
- GPU execution time and result
- Performance comparison (speedup/slowdown)
- Result verification (ensures CPU and GPU results match)

Typical results show significant GPU acceleration for large datasets, though the exact speedup depends on your hardware.

## 🏗️ Project Structure

```
metal-rs/
├── src/
│   ├── main.rs          # Main application logic
│   ├── constants.rs     # Configuration constants
│   └── metal/
│       └── sum.metal    # Metal shader for GPU computation
├── build/               # Compiled Metal library (generated)
├── Cargo.toml          # Rust dependencies
├── Makefile            # Build automation
└── README.md           # This file
```

## 🔧 Configuration

Key parameters can be modified in `src/constants.rs`:
- `LIST_SIZE`: Total number of elements to process (default: 1 billion)
- `CHUNK_SIZE`: Elements per chunk (default: 100 million)
- `NUM_CHUNKS`: Number of chunks (default: 10)
- `THREAD_GROUP_SIZE`: GPU threads per group (default: 32)