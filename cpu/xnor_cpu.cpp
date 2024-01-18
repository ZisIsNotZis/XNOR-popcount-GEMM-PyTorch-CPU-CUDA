/*
    PyTorch-XNOR-GEMM-Extention
    Authors: Tairen (tairenpiao@gmail.com)
    This code can only be used for research purposes.
    For other purposes (e.g., commercial), please contact me.
*/

#include <iostream>
#include <typeinfo>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/AccumulateType.h>

#define BIT 64
#define TYPE int64_t
#define UTYPE uint64_t
#define DTYPE torch::kI64
#define POPCOUNT __builtin_popcountll

at::Tensor pak(at::Tensor &olda) {
    auto a = olda.unflatten(-1, {-1,BIT});
    auto out = at::empty(a.index({"...",0}).sizes(), DTYPE);
    at::TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .is_reduction(true)
        .declare_static_shape(a.sizes())
        .add_owned_output(out)
        .add_owned_input(a)
        .build()
        .for_each([](char **dat, const long *stride, long n){
            UTYPE r = 0;
            for(long i=0; i<n; i++) {
                r |= (UTYPE)*(unsigned char*)(dat[1]+(i+1)*stride[1]-1) >> 7 << i;
            }
            *(UTYPE*)dat[0] = r;
        });
    return out;
}

at::Tensor mm_f(at::Tensor &a, at::Tensor &b) {
    auto size = at::broadcast_tensors({a, b})[0].sizes();
    auto out = at::empty(std::vector<long>(size.begin(), size.end()-1), torch::kF32);
    at::TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .is_reduction(true)
        .declare_static_shape(size)
        .add_owned_output(out)
        .add_owned_input(a)
        .add_owned_input(b)
        .build()
        .for_each([](char **dat, const long *stride, long n){
            long t = 0;
            for(long i=0; i<n; i++) {
                t += POPCOUNT(*(UTYPE*)(dat[1]+i*stride[1]) ^ *(UTYPE*)(dat[2]+i*stride[2]));
            }
            *(float*)dat[0] = n * BIT - t * 2;
        });
    return out;
}

at::Tensor mm_i1(at::Tensor &a, at::Tensor &b) {
    auto size = at::broadcast_tensors({a, b})[0].sizes();
    auto out = at::empty(std::vector<long>(size.begin(), size.end()-1), torch::kI8);
    at::TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .is_reduction(true)
        .declare_static_shape(size)
        .add_owned_output(out)
        .add_owned_input(a)
        .add_owned_input(b)
        .build()
        .for_each([](char **dat, const long *stride, long n){
            char t = 0;
            for(long i=0; i<n; i++) {
                t += POPCOUNT(*(UTYPE*)(dat[1]+i*stride[1]) ^ *(UTYPE*)(dat[2]+i*stride[2]));
            }
            *(char*)dat[0] = n * BIT - t * 2;
        });
    return out;
}

at::Tensor mm(at::Tensor &a, at::Tensor &b) {
    at::Tensor t = mm_i1(a, b);
    return pak(t);
}

at::Tensor mm_cont(at::Tensor &olda, at::Tensor &oldb) {
    auto a = olda.unflatten(-2, {-1,BIT}).flatten(-2);
    auto b = oldb.unflatten(-2, {-1,BIT}).flatten(-2);
    auto size = at::broadcast_tensors({a, b})[0].sizes();
    auto out = at::empty(std::vector<long>(size.begin(), size.end()-1), torch::kI8);
    at::TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .is_reduction(true)
        .declare_static_shape(size)
        .add_owned_output(out)
        .add_owned_input(a)
        .add_owned_input(b)
        .build()
        .for_each([](char **dat, const long *stride, long n){
            UTYPE r = 0;
            for(int i=0; i<BIT; i++) {
                UTYPE t = 0;
                for(long j=0; j<n / BIT; j++) {
                    t += POPCOUNT(*(UTYPE*)(dat[1]+(i*n/BIT+j)*stride[1]) ^ *(UTYPE*)(dat[2]+(i*n/BIT+j)*stride[2]));
                }
                r |= t >> 63 << i;
            }
            *(UTYPE*)dat[0] = r;
        });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pak", &pak, "pak");
    m.def("mm_f", &mm_f, "mm_f");
    m.def("mm_i1", &mm_i1, "mm_i1");
    m.def("mm", &mm, "mm");
    m.def("mm_cont", &mm, "mm_cont");
}