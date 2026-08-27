#include "numpy.h"
#include "internel.h"

#define back(k)\
if (back_status!=NULL){\
	*back_status=k;\
}

void ndarray_matmul(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	//检查传入指针
	if (in1==NULL || in1->base==NULL || in2==NULL || in2->base==NULL || out==NULL){
		back(NDARRAY_ERR_NULLPTR);
		return;
	}
	
	bool can_b=_can_broadcast(in1,in2,true);
	if (can_b==false){
		back(NDARRAY_ERR_CANNOT_BROADCAST);
		return;
	}
	
	//大分支，out是否为空
	if (out->base==NULL){
		//需要的指针
		storage *new_storage=NULL;//48
		double *new_data=NULL;//49
		uint64_t *new_shape=NULL;//35
		uint64_t *new_stride=NULL;//36
		
		
		uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
		//分配shape与stride
		new_shape=(uint64_t*)malloc(out_dim*sizeof(uint64_t));
		new_stride=(uint64_t*)malloc(out_dim*sizeof(uint64_t));
		if (new_shape==NULL || new_stride==NULL){
			free(new_storage);
			free(new_shape);
			free(new_stride);
			free(new_data);
			back(NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		_calculate_shape(in1,in2,new_shape,true);
		
		//计算总数与连续步长
		uint64_t total;
		_shape_to_stride_total(out_dim,new_shape,&total,new_stride);
		
		//分配storage与data
		new_storage=(storage*)malloc(sizeof(storage));
		new_data=(double*)malloc(total*sizeof(double));
		if (new_storage==NULL || new_data==NULL){
			free(new_storage);
			free(new_shape);
			free(new_stride);
			free(new_data);
			back(NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		
		//挂载
		out->ndim=out_dim;
		out->shape=new_shape;
		out->stride=new_stride;
		out->offset=0;
		out->total_num=total;
		out->base=new_storage;
		new_storage->refer_count=1;
		new_storage->total_num=total;
		new_storage->data=new_data;
		
	}
	else{
		uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
		if (out->ndim!=out_dim){
			back(NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		
		uint64_t new_shape[UINT8_MAX];
		_calculate_shape(in1,in2,new_shape,true);
		
		for (uint8_t i=0;i<out_dim;i++){
			if (new_shape[i]!=out->shape[i]){
				back(NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
	}
	
	//核心计算逻辑
	uint8_t b1 = in1->ndim - 2;
	uint8_t b2 = in2->ndim - 2;
	uint8_t out_dim = in1->ndim > in2->ndim ? in1->ndim : in2->ndim;
	uint8_t out_b = out_dim - 2;
	uint8_t start1 = out_b - b1;
	uint8_t start2 = out_b - b2;
	uint64_t k = in1->shape[b1 + 1];
	uint64_t coords_out[UINT8_MAX], coords1[UINT8_MAX], coords2[UINT8_MAX];
	
	for (uint64_t linear_idx = 0; linear_idx < out->total_num; linear_idx++) {
		_linear_to_coords(linear_idx, out->ndim, out->shape, coords_out);
		uint64_t M = coords_out[out_b];
		uint64_t N = coords_out[out_b + 1];
		double sum = 0.0;
		for (uint64_t k_idx = 0; k_idx < k; k_idx++) {
			// in1 坐标
			for (uint8_t d = 0; d < b1; d++) {
				coords1[d] = (in1->shape[d] == 1) ? 0 : coords_out[start1 + d];
			}
			coords1[b1] = M;
			coords1[b1 + 1] = k_idx;
			// in2 坐标
			for (uint8_t d = 0; d < b2; d++) {
				coords2[d] = (in2->shape[d] == 1) ? 0 : coords_out[start2 + d];
			}
			coords2[b2] = k_idx;
			coords2[b2 + 1] = N;
			// 线性索引并累乘
			uint64_t idx1 = _coords_to_linear(coords1, in1->ndim, in1->offset, in1->stride);
			uint64_t idx2 = _coords_to_linear(coords2, in2->ndim, in2->offset, in2->stride);
			sum += in1->base->data[idx1] * in2->base->data[idx2];
		}
		uint64_t out_idx = _coords_to_linear(coords_out, out->ndim, out->offset, out->stride);
		out->base->data[out_idx] = sum;
	}
	back(NDARRAY_OK);
	return;
}
