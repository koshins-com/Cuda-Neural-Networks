source=./src/*.cu
FLAGS = -lcurand -Iinclude

build/main: $(source)
	mkdir -p build
	nvcc $(FLAGS) $(source) -o build/main

run: build/main
	./build/main

build/debug: $(source)
	mkdir -p build
	nvcc $(FLAGS) $(source) -G -g -o build/debug

debug: build/debug
	cuda-gdb build/debug -ex "lay src"
