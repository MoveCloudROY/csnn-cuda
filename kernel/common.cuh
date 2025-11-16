#pragma once

#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_runtime.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

