#include "numpy.h"
#include "internel.h"

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

static void _calculate_func(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t type,uint8_t *back_status){
	//检查传入指针
	if (in1==NULL || in2==NULL || in1->base==NULL || in2->base==NULL || out==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return;
	}
	
	//判断能否广播
	bool can_b=_can_broadcast(in1,in2,false);
	if (can_b==false){
		//无法广播
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_CANNOT_BROADCAST;
		}
		return;
	}
	
	uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
	uint64_t new_shape[UINT8_MAX];
	uint64_t new_stride1[UINT8_MAX];
	uint64_t new_stride2[UINT8_MAX];
	uint64_t out_stride[UINT8_MAX];
	
	_calculate_shape(in1,in2,new_shape,false);
	_calculate_broadcast(in1,in2,new_stride1,new_stride2,false);
	
	//计算连续步长
	for (uint8_t i1=0;i1<out_dim-1;i1++){
		uint64_t curr=1;
		for (uint8_t i2=i1+1;i2<out_dim-1;i2++){
			curr*=new_shape[i2];
		}
		out_stride[i1]=curr;
	}
	out_stride[out_dim-1]=1;
	
	//计算总元素量
	uint64_t total=1;
	for (uint8_t i=0;i<out_dim;i++){
		total*=new_shape[i];
	}
	
	//检查out是否为空
	if (out->base==NULL){
		//需要的指针
		
	}
	else{
		
	}
}

void ndarray_add(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	
}

void ndarray_sub(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	
}

void ndarray_mul(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	
}

void ndarray_div(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	
}

void ndarray_apply(const ndarray *in,double (*apply_func)(double,void*),void *input,ndarray *out,uint8_t *back_status){
	
}
