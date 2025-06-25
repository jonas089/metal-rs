all: compile

compile:
	xcrun -sdk macosx metal src/metal/sum.metal -o build/sum.metallib
run:
	cargo run --release