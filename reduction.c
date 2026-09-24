#include "numpy.h"
#include "internel.h"

static inline void _back(uint8_t *back_status,uint8_t kind){
	if (back_status!=NULL){
		*back_status=kind;
	}
	return;
}

static inline void _check(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,uint8_t *back_status){
	if (dim_num>in->ndim || dim_num==0){
		_back(back_status,NDARRAY_ERR_WRONGDIM);
		return;
	}
	
	for (uint8_t i=0;i<dim_num;i++){
		if (dim_list[i]>=in->ndim){
			_back(back_status,NDARRAY_ERR_DIM_OUT_OF_RANGE);
			return;
		}
	}
	
	bool visited[UINT8_MAX]={false};
	for (uint8_t i=0;i<dim_num;i++){
		uint8_t dim=dim_list[i];
		if (visited[dim]==false){
			visited[dim]=true;
		}
		else{
			_back(back_status,NDARRAY_ERR_DIM_REPEAT);
			return;
		}
	}
	
	_back(back_status,NDARRAY_OK);
	return;
}

void ndarray_sum(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_mean(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_variance(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_max(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_min(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

