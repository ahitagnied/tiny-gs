# Usage: source ./source.sh
# Activates the project venv and points the build toolchain at:
#   - CUDA 12.8 (matches torch's cu128 build)
#   - gcc/g++ 12 (CUDA 12.8 rejects host gcc > 14, and the default conda gcc is 15)
# Without this, gsplat's JIT compile picks the wrong nvcc, can't find
# cuda_runtime.h, or trips host_config.h's gcc-version guard.

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
