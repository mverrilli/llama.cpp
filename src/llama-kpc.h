#pragma once
#include "ggml.h"

// short aliases for the GGML_KPC_* wire-format constants declared in ggml.h
#define KPC_GROUP          GGML_KPC_GROUP            // token group size
#define KPC_SZ_GROUP_BYTES GGML_KPC_SZ_GROUP_BYTES   // int8 scale/zp slab bytes per group
