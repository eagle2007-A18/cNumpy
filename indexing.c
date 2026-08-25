#include "numpy.h"

void get_number(const ndarray *in,uint64_t *index,double *ans,uint8_t *back_status){
	if (in==NULL || in->base==NULL || ans==NULL || index==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		ans=NULL;
		return;
	}
	
	//标量
	if (in->shape==NULL){
		if (index[0]!=0){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_INDEX_OUT_OF_RANGE;
			}
			ans=NULL;
			return;
		}
		else{
			*ans=in->base->data[0];
			if (back_status!=NULL){
				*back_status=NDARRAY_OK;
			}
			return;
		}
	}
	
	//检查index
	for (uint8_t i=0;i<in->ndim;i++){
		if (index[i]>=in->shape[i]){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_INDEX_OUT_OF_RANGE;
			}
			ans=NULL;
			return;//索引超限
		}
	}
	
	uint64_t curr_offset=in->offset;
	for (uint8_t i=0;i<in->ndim;i++){
		curr_offset+=in->stride[i]*index[i];
	}
	*ans=in->base->data[curr_offset];
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return;
}

void change_number(ndarray *in,uint64_t *index,double new_number,uint8_t *back_status){
	//传入指针检查
	if (in==NULL || in->base==NULL || index==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return;
	}
	
	//标量
	if (in->shape==NULL){
		if (index[0]!=0){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_INDEX_OUT_OF_RANGE;
			}
			return;
		}
		else{
			in->base->data[0]=new_number;
			if (back_status!=NULL){
				*back_status=NDARRAY_OK;
			}
			return;
		}
	}
	
	//索引超限检查
	for (uint8_t i=0;i<in->ndim;i++){
		if (index[i]>=in->shape[i]){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_INDEX_OUT_OF_RANGE;
			}
			return;
		}
	}
	
	uint64_t curr_offset=in->offset;
	for (uint8_t i=0;i<in->ndim;i++){
		curr_offset+=in->stride[i]*index[i];
	}
	in->base->data[curr_offset]=new_number;
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return;
}

void slice(const ndarray *in,Slice_index *slice_list,ndarray *out,uint8_t *back_status){
	//传入指针检查
	if (in==NULL || in->base==NULL || slice_list==NULL || out==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return;
	}
	
	//检查是否为标量
	if (in->ndim==0){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_SCALAR_CANNOT_SLICE;
		}
		return;//标量不可切片
	}
	
	//检查out是否初始化
	if (out->base!=NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_OUT_IS_INIT;
		}
		return;
	}
	
	uint8_t i=0;
	uint64_t curr_offset=in->offset;
	
	//临时数组
	uint64_t new_shape[UINT8_MAX];
	uint64_t new_stride[UINT8_MAX];
	
	for (i=0;i<in->ndim;i++){
		Slice_index s=slice_list[i];
		uint64_t max=in->shape[i];
		
		if (s.step<=0){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_STEP_IS_NAT;
			}
			return;//切片步长为负数或0
		}
		
		if (s.end==0 || s.start>=max || s.end>max || s.start>=s.end){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_SLICE_OUT_OF_RANGE;
			}
			return;//slice索引超限
		}
		
		uint8_t len=(s.end-s.start+s.step-1)/s.step;
		uint64_t new_stride_i=in->stride[i]*s.step;
		curr_offset+=in->stride[i]*s.start;
		new_shape[i]=len;
		new_stride[i]=new_stride_i;
	}
	
	//分配内存
	out->shape=(uint64_t*)malloc(in->ndim*sizeof(uint64_t));
	out->stride=(uint64_t*)malloc(in->ndim*sizeof(uint64_t));
	
	if (out->shape==NULL || (*out).stride==NULL){
		if (out->shape==NULL){
			free(out->stride);
		}
		else{
			free(out->shape);
		}
		
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_ALLOC_FAIL;
		}
		return;//内存分配失败
	}
	
	//填充shape和stride
	for (uint8_t i=0;i<in->ndim;i++){
		out->shape[i]=new_shape[i];
		out->stride[i]=new_stride[i];
	}
	
	//计算总量
	uint64_t total=1;
	for (uint8_t i=0;i<in->ndim;i++){
		total*=new_shape[i];
	}
	
	out->total_num=total;
	out->ndim=in->ndim;
	out->offset=curr_offset;
	out->base=in->base;
	in->base->refer_count+=1;
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return;
}
