#ifndef BASIC_H
#define BASIC_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

//错误码
//基础部分 1-20
#define NDARRAY_OK 0//成功
#define NDARRAY_ERR_NULLPTR 1//传入指针为空
#define NDARRAY_ERR_ALLOC_FAIL 2//内存分配失败

//creation部分错误码 20-40
#define NDARRAY_ERR_ZERO_SHAPE 21//创建时传入的shape数组里面包含0
#define NDARRAY_ERR_EMPTY_VIEW 22//空ndarray不可拥有视图

//indexing部分错误码 40-60
#define NDARRAY_ERR_SLICE_OUT_OF_RANGE 41//slice超限
#define NDARRAY_ERR_INDEX_OUT_OF_RANGE 42//索引超限
#define NDARRAY_ERR_SCALAR_CANNOT_SLICE 43//标量不可切片
#define NDARRAY_ERR_STEP_IS_NAT 44//切片步长不可为负数和0
//#define NDARRAY_ERR_END_BIG_START 8//切片start小于end
#define NDARRAY_ERR_OUT_IS_INIT 45//slice函数的out已经初始化

//calculate部分错误码 60-80
#define NDARRAY_ERR_CANNOT_BROADCAST 60
#define NDARRAY_ERR_WRONG_SHAPE 61

//reduction部分错误码 80-100
#define NDARRAY_ERR_WRONGDIM 81//dim必须大于0
#define NDARRAY_ERR_DIM_OUT_OF_RANGE 82
#define NDARRAY_ERR_DIM_REPEAT 83

typedef struct storage{
	double* data;
	uint64_t refer_count;
	uint64_t total_num;
}storage;

typedef struct ndarray{
	uint8_t ndim;
	uint64_t *shape;
	uint64_t *stride;
	uint64_t offset;
	uint64_t total_num;
	storage *base;
}ndarray;
/*
当ndarray是标量时：
ndim=0  shape=NULL  stride=NULL total_size=1  
*/

typedef struct Slice_index{
	uint8_t start;
	uint8_t end;
	uint8_t step;
}Slice_index;

#endif
