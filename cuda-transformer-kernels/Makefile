# Default arch = sm_75 (Tesla T4 on Colab/Kaggle). Override: make ARCH=sm_80
ARCH  ?= sm_75
NVCC  ?= nvcc
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: bin/bench_gemm bin/bench_rownorm bin/bench_attention

bin/bench_gemm: src/bench_gemm.cu src/common.cuh src/gemm_kernels.cuh
	@mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

bin/bench_rownorm: src/bench_rownorm.cu src/common.cuh src/reduce.cuh src/rownorm_kernels.cuh
	@mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@

bin/bench_attention: src/bench_attention.cu src/common.cuh src/reduce.cuh src/attention_kernels.cuh
	@mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

# Part 1: SGEMM
run: bin/bench_gemm
	@mkdir -p results
	./bin/bench_gemm --csv results/gemm.csv

plot:
	python3 scripts/plot_gemm.py results/gemm.csv

# Part 2: softmax + LayerNorm
run-rownorm: bin/bench_rownorm
	@mkdir -p results
	./bin/bench_rownorm --csv results/rownorm.csv

torch-baseline:
	@mkdir -p results
	python3 scripts/torch_baseline.py --csv results/torch_rownorm.csv

plot-rownorm:
	python3 scripts/plot_rownorm.py results/rownorm.csv results/torch_rownorm.csv

# Part 3: attention
run-attention: bin/bench_attention
	@mkdir -p results
	./bin/bench_attention --csv results/attention.csv

torch-attention:
	@mkdir -p results
	python3 scripts/torch_attention.py --csv results/torch_attention.csv

plot-attention:
	python3 scripts/plot_attention.py results/attention.csv results/torch_attention.csv

# Part 4: PyTorch extension (built on first import, ~1-2 min)
ext-deps:
	pip install -q ninja pytest

ext-test: ext-deps
	python3 -m pytest -q tests

ext-bench: ext-deps
	@mkdir -p results
	python3 scripts/bench_ext.py --csv results/ext_bench.csv

clean:
	rm -rf bin

.PHONY: all run plot run-rownorm torch-baseline plot-rownorm run-attention torch-attention plot-attention ext-deps ext-test ext-bench clean
