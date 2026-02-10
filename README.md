# det-matmul

Implementation for deterministic matrix multiplication across heterogeneous architectures

## Experimentation

```bash
git clone https://github.com/sanjibansg/det-matmul.git
cd det-matmul/
git submodule update --init --recursive
mkdir build && cd build
cmake -DENABLE_CUDA=ON ..
cmake --build .
./DetMatMul 
```
To experiment only on CPUs, we need the -DENABLE_CUDA=OFF option.

## License

[MIT](https://github.com/sanjibansg/det-matmul/blob/main/LICENSE)