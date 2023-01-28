/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The CUDA kernel is modified from SplitGelu plugin of TensorRT 8.5
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/diffusion/split_gelu_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int32_t HHS, int32_t TPB>
__global__ void splitGeluKernel(T const* input, T* output) {
  int32_t index_input = blockIdx.x * HHS * 2 + threadIdx.x;
  int32_t index_output = blockIdx.x * HHS + threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < HHS / TPB; ++i) {
    auto value_left = static_cast<float>(input[index_input]);
    auto value_right = static_cast<float>(input[index_input + HHS]);

    // Gelu is applied to right side only: Gelu(x) = x * 0.5 * (erf(x / 1.41421356237) + 1.0)
    float gelu_right = value_right * 0.5f * (erff(value_right / 1.41421356237f) + 1.0f);
    float result = value_left * gelu_right;
    output[index_output] = static_cast<T>(result);
    index_input += TPB;
    index_output += TPB;
  }
  return;
}

template <typename T>
void LaunchSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size, T const* input, T* output) {
  constexpr int32_t TPB = 256;  // thread per block
  switch (half_hidden_size) {
    case 1280:
      (splitGeluKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 2560:
      (splitGeluKernel<T, 2560, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 5120:
      (splitGeluKernel<T, 5120, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template __global__ void splitGeluKernel<float, 1280, 256>(float const*, float*);
template __global__ void splitGeluKernel<float, 2560, 256>(float const*, float*);
template __global__ void splitGeluKernel<float, 5120, 256>(float const*, float*);
template __global__ void splitGeluKernel<half, 1280, 256>(half const*, half*);
template __global__ void splitGeluKernel<half, 2560, 256>(half const*, half*);
template __global__ void splitGeluKernel<half, 5120, 256>(half const*, half*);

template void LaunchSplitGeluKernel<float>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                                           float const* input, float* output);

template void LaunchSplitGeluKernel<half>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                                          half const* input, half* output);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
