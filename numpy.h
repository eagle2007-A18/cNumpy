#ifndef NUMPY_H
#define NUMPY_H

#include "basic.h"
#include "internel.h"

//ndarray创建与清理函数
ndarray* ndarray_init_empty(uint8_t *back_status);
//创建空ndarray，内部指针全部置空

ndarray* ndarray_init_full(uint8_t ndim,uint64_t *shape,double full_num,uint8_t *back_status);
//创建ndarray，填充full_num

ndarray* ndarray_init_random_uniform(uint8_t ndim,uint64_t *shape,double start,double end,uint8_t *back_status);
//创建ndarray，填充在[start,end)之间的均匀分布随机数

ndarray* ndarray_init_random_nature(uint8_t ndim,uint64_t *shape,double mean,double variance,uint8_t *back_status);
//创建ndarray，填充均值为mean，方差为variance的正态分布随机数

ndarray* ndarray_copy(const ndarray *from,uint8_t *back_status);
//深拷贝from，在此过程里把内存变成连续

ndarray* ndarray_view(const ndarray *father,uint8_t *back_status);
//建立father下的storage的视图

void ndarray_free(ndarray *in);
//彻底释放，清理空间

void ndarray_empty(ndarray *in);
//把in清空


//索引函数
void get_number(const ndarray *in,uint64_t *index,double *ans,uint8_t *back_status);
//获取特定位置的值

void change_number(ndarray *in,uint64_t *index,double new_number,uint8_t *back_status);
//修改特定位置的值

void slice(const ndarray *in,Slice_index *slice_list,ndarray *out,uint8_t *back_status);
//获取切片


//运算函数
void ndarray_add(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status);//加
void ndarray_sub(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status);//减
void ndarray_mul(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status);//乘
void ndarray_div(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status);//除

void ndarray_matmul(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status);//矩阵乘法

void ndarray_apply(const ndarray *in,double (*apply_func)(double,void*),void *input,ndarray *out,uint8_t *back_status);
//针对in里面的每一个元素应用apply_func，写入out里


//形状操作函数
void ndarray_reshape(const ndarray *in,uint8_t new_ndim,uint64_t *new_shape,ndarray *out,uint8_t *back_status);//变形
void ndarray_swapaxes(const ndarray *in,uint8_t dim1,uint8_t dim2,ndarray *out,uint8_t *back_status);//转置
void ndarray_flatten(const ndarray *in,ndarray *out,uint8_t *back_status);//展平
void ndarray_squeeze(const ndarray *in,ndarray *out,uint8_t dim,bool is_auto,uint8_t *back_status);
//压缩维度，如果is_auto=true，压缩所有dim=1的维度
void ndarray_unsqueeze(const ndarray *in,ndarray *out,uint8_t dim,uint8_t *back_status);
//在dim处插入为1的维度


//拼接
void ndarray_cat(const ndarray *in_list,uint64_t cat_number,uint8_t cat_dim,ndarray *out,uint8_t *back_status);
//把cat_number个ndarray在cat_dim上拼接起来

//规约函数
void ndarray_sum(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status);
void ndarray_mean(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status);
void ndarray_variance(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status);
void ndarray_max(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status);
void ndarray_min(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status);

#endif
