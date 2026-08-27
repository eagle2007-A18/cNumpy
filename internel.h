#ifndef INTERNEL_H
#define INTERNEL_H

#include "basic.h"

bool _is_continuous(const ndarray *in);

bool _can_broadcast(const ndarray *in1,const ndarray *in2,bool is_matmul);

bool _calculate_broadcast(const ndarray *in1,const ndarray *in2,uint64_t *new_stride1,uint64_t *new_stride2,bool is_matmul);

bool _calculate_shape(const ndarray *in1,const ndarray *in2,uint64_t *new_shape,bool is_matmul);

void _linear_to_coords(uint64_t linear_idx,uint8_t ndim,const uint64_t *shape,uint64_t *out);

uint64_t _coords_to_linear(const uint64_t *in_coords,uint8_t ndim,uint64_t offset,const uint64_t *in_stride);

void _shape_to_stride_total(uint8_t ndim,const uint64_t *shape,uint64_t *out_total,uint64_t *out_stride);

#endif
