#include "numpy.h"
#include "internel.h"

static inline void _back(uint8_t *back_status,uint8_t kind){
	if (back_status!=NULL){
		*back_status=kind;
	}
	return;
}

void ndarray_reshape(const ndarray *in,uint8_t new_ndim,uint64_t *new_shape,ndarray *out,uint8_t *back_status){
	//检查传入指针
	if (in==NULL || new_shape==NULL || out==NULL){
		_back(back_status,NDARRAY_ERR_NULLPTR);
		return;
	}
	
	if (new_ndim==0){
		_back(back_status,NDARRAY_ERR_WRONGDIM);
		return;
	}
	
	
}

void ndarray_swapaxes(const ndarray *in,uint8_t dim1,uint8_t dim2,ndarray *out,uint8_t *back_status){
	
}

void ndarray_flatten(const ndarray *in,ndarray *out,uint8_t *back_status){
	
}

void ndarray_squeeze(const ndarray *in,ndarray *out,uint8_t dim,bool is_auto,uint8_t *back_status){
	
}

void ndarray_unsqueeze(const ndarray *in,ndarray *out,uint8_t dim,uint8_t *back_status){
	
}

void ndarray_cat(const ndarray *in_list,uint64_t cat_number,uint8_t cat_dim,ndarray *out,uint8_t *back_status){
	
}
