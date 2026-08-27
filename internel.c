#include "internel.h"

bool _is_continuous(const ndarray *in){
	if (in->base==NULL){
		//空张量
		return false;
	}
	
	if (in->ndim==0){
		//标量默认连续
		return true;
	}
	
	uint64_t curr=1;
	for (int16_t i=(int16_t)in->ndim-1;i>=0;i--){
		if (in->stride[i]!=curr){
			return false;
		}
		curr*=in->shape[i];
	}
	return true;
}

bool _can_broadcast(const ndarray *in1,const ndarray *in2,bool is_matmul){
	if (in1->base==NULL || in2->base==NULL){
		return false;
	}
	
	if (is_matmul==true){
		if (in1->ndim<2 || in2->ndim<2){
			//标量和一维向量不能进行矩阵乘法
			return false;
		}
		
		uint64_t k1=in1->shape[in1->ndim-1];
		uint64_t k2=in2->shape[in2->ndim-2];
		if (k1!=k2){
			//最后两维不相等
			return false;
		}
		
		int16_t i=(int16_t)in1->ndim-3;
		int16_t j=(int16_t)in2->ndim-3;
		
		while (i>=0 || j>=0){
			uint64_t dim1=i>=0?in1->shape[i]:1;
			uint64_t dim2=j>=0?in2->shape[j]:1;
			if (dim1==dim2 || dim1==1 || dim2==1){
				i--;
				j--;
			}
			else{
				return false;
			}
		}
		return true;
	}
	else{
		if (in1->ndim==0 || in2->ndim==0){
			//标量默认可以广播
			return true;
		}
		
		int16_t i=(int16_t)in1->ndim-1;
		int16_t j=(int16_t)in2->ndim-1;
		
		while (i>=0 || j>=0){
			uint64_t dim1=i>=0?in1->shape[i]:1;
			uint64_t dim2=j>=0?in2->shape[j]:1;
			
			if (dim1==dim2 || dim1==1 || dim2==1){
				i--;
				j--;
			}
			else{
				return false;
			}
		}
		return true;
	}
}

bool _calculate_broadcast(const ndarray *in1,const ndarray *in2,uint64_t *new_stride1,uint64_t *new_stride2,bool is_matmul){
	if (_can_broadcast(in1,in2,is_matmul)==false){
		return false;
	}
	
	uint8_t ndim1=in1->ndim;
	uint8_t ndim2=in2->ndim;
	
	uint64_t *shape1=in1->shape;
	uint64_t *shape2=in2->shape;
	
	uint64_t *stride1=in1->stride;
	uint64_t *stride2=in2->stride;
	
	if (is_matmul==true){
		int16_t b1=(int16_t)ndim1-2;
		int16_t b2=(int16_t)ndim2-2;
		int16_t out_b=b1>b2?b1:b2;
		int16_t i=b1-1;
		int16_t j=b2-1;
		int16_t k=out_b-1;
		
		while (i>=0 || j>=0){
			uint64_t dim1=i>=0?shape1[i]:1;
			uint64_t dim2=j>=0?shape2[j]:1;
			
			new_stride1[k]=dim1==1?0:stride1[i];
			new_stride2[k]=dim2==1?0:stride2[j];
			i--;
			j--;
			k--;
		}
		
		new_stride1[out_b]=stride1[ndim1-2];
		new_stride1[out_b+1]=0;
		new_stride2[out_b]=0;
		new_stride2[out_b+1]=stride2[ndim2-1];
		return true;
	}
	else{
		uint64_t out_ndim=ndim1>ndim2?ndim1:ndim2;
		int16_t i=(int16_t)ndim1-1;
		int16_t j=(int16_t)ndim2-1;
		int16_t k=(int16_t)out_ndim-1;
		
		while (i>=0 || j>=0){
			uint64_t dim1=i>=0?shape1[i]:1;
			uint64_t dim2=j>=0?shape2[j]:1;
			//uint64_t out_dim=dim1>dim2?dim1:dim2;
			
			new_stride1[k]=dim1==1?0:stride1[i];
			new_stride2[k]=dim2==1?0:stride2[j];
			
			i--;
			j--;
			k--;
		}
		return true;
	}
}

bool _calculate_shape(const ndarray *in1,const ndarray *in2,uint64_t *new_shape,bool is_matmul){
	//传入指针检查
	if (in1==NULL || in2==NULL || new_shape==NULL){
		return false;
	}
	
	//判断能否广播
	bool can_broadcast=_can_broadcast(in1,in2,is_matmul);
	if (can_broadcast==false){
		return false;
	}
	
	if (is_matmul==true){
		//矩阵乘法
		uint8_t b1=in1->ndim-2;
		uint8_t b2=in2->ndim-2;
		uint8_t out_b=b1>b2?b1:b2;
		int16_t i=(int16_t)b1-1;
		int16_t j=(int16_t)b2-1;
		int16_t k=(int16_t)out_b-1;
		while (k>=0){
			uint64_t dim1=(i>=0)?in1->shape[i]:1;
			uint64_t dim2=(j>=0)?in2->shape[j]:1;
			new_shape[k]=dim1>dim2?dim1:dim2;
			i--;
			j--;
			k--;
		}
		new_shape[out_b]=in1->shape[in1->ndim-2];
		new_shape[out_b+1]=in2->shape[in2->ndim-1];
		return true;
	}
	else{
		//非矩阵乘法
		uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
		int16_t i=(int16_t)in1->ndim-1;
		int16_t j=(int16_t)in2->ndim-1;
		int16_t k=(int16_t)out_dim-1;
		
		while (k>=0){
			uint64_t dim1=(i>=0)?in1->shape[i]:1;
			uint64_t dim2=(j>=0)?in2->shape[j]:1;
			new_shape[k]=dim1>dim2?dim1:dim2;
			i--;
			j--;
			k--;
		}
		return true;
	}
}

void _linear_to_coords(uint64_t linear_idx,uint8_t ndim,const uint64_t *shape,uint64_t *out){
	if (ndim==0){
		return;
	}
	for (int16_t d=(int16_t)ndim-1;d>=0;d--){
		out[d]=linear_idx%shape[d];
		linear_idx/=shape[d];
	}
}

uint64_t _coords_to_linear(const uint64_t *in_coords,uint8_t ndim,uint64_t offset,const uint64_t *in_stride){
	uint64_t curr=offset;
	if (ndim==0){
		return curr;
	}
	else{
		for (uint8_t i=0;i<ndim;i++){
			curr+=in_coords[i]*in_stride[i];
		}
		return curr;
	}
}
