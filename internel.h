#ifndef INTERNEL_H
#define INTERNEL_H

#include "basic.h"

bool _is_continuous(const ndarray *in);

bool _can_broadcast(const ndarray *in1,const ndarray *in2,bool is_matmul);

bool _calculate_broadcast(const ndarray *in1,const ndarray *in2,uint64_t *new_stride1,uint64_t *new_stride2,bool is_matmul);

bool _calculate_shape(const ndarray *in1,const ndarray *in2,uint64_t *new_shape,bool is_matmul);

#endif
