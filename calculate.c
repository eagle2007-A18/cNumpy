#include "numpy.h"
#include "internel.h"

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

static void _calculate_func(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t type,uint8_t *back_status){
	//检查传入指针
	if (in1==NULL || in1->base==NULL || in2==NULL || in2->base==NULL || out==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return;
	}
	
	if (out->base==NULL){
		//内部计算，分配内存
		bool can_c=_can_broadcast(in1,in2,false);
		if (can_c==false){
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_CANNOT_BROADCAST;
			}
			return;
		}
		uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
		//需要使用的指针
		
		
	}
	else{
		//先检查，再填充
		
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
