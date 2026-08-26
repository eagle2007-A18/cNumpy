#include "numpy.h"
#include "internel.h"

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

#define back(k)\
if (back_status!=NULL){\
	*back_status=k;\
}

static void _calculate_func(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t type,uint8_t *back_status){
	//检查传入指针
	/*
	in1,in2必须初始化
	out不一定
	*/
	if (in1==NULL || in2==NULL || in1->base==NULL || in2->base==NULL || out==NULL){
		back(NDARRAY_ERR_NULLPTR);
		return;
	}
	
	//检查能否进行广播
	if (_can_broadcast(in1,in2,false)==false){
		back(NDARRAY_ERR_CANNOT_BROADCAST);
		return;
	}
	
	//分情况，out是否进行分配
	if (out->base==NULL){
		//未分配
		//使用的指针
		storage *new_storage=NULL;
		double *new_data=NULL;
		uint64_t *new_shape=NULL;
		uint64_t *new_stride=NULL;
		
		//标量和标量运算
		if (in1->ndim==0 && in2->ndim==0){
			//成立
			new_storage=(storage*)malloc(sizeof(storage));
			new_data=(double*)malloc(sizeof(double));
			if (new_storage==NULL || new_data==NULL){
				goto cleanup1;//内存分配失败
			}
			
			//out初始化
			out->ndim=0;
			out->total_num=1;
			out->offset=0;
			out->shape=new_shape;
			out->stride=new_stride;
			out->base=new_storage;
			new_storage->data=new_data;
			new_storage->refer_count=1;
			new_storage->total_num=1;
			switch (type) {
			case ADD:
				new_data[0]=in1->base->data[0]+in2->base->data[0];
				break;
			case SUB:
				new_data[0]=in1->base->data[0]-in2->base->data[0];
				break;
			case MUL:
				new_data[0]=in1->base->data[0]*in2->base->data[0];
				break;
			case DIV:
				new_data[0]=in1->base->data[0]/in2->base->data[0];
				break;
			default:
				//TODO
				break;
			}
			back(NDARRAY_OK);
			return;
		}
		else{
			//不成立
			uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
			
			//计算
			uint64_t new_stride1[UINT8_MAX];
			uint64_t new_stride2[UINT8_MAX];
			new_shape=(uint64_t*)malloc(out_dim*sizeof(uint64_t));
			new_stride=(uint64_t*)malloc(out_dim*sizeof(uint64_t));
			if (new_shape==NULL || new_stride==NULL){
				goto cleanup1;
			}
			
			_calculate_broadcast(in1,in2,new_stride1,new_stride2,false);
			_calculate_shape(in1,in2,new_shape,false);
			
			//计算输出总元素数和连续步长
			for (uint8_t i1=0;i1<out_dim-1;i1++){
				uint64_t curr=1;
				for (uint8_t i2=i1+1;i2<out_dim;i2++){
					curr*=new_shape[i2];
				}
				new_stride[i1]=curr;
			}
			new_stride[out_dim-1]=1;
			
			uint64_t total=1;
			for (uint8_t i=0;i<out_dim;i++){
				total*=new_shape[i];
			}
			
			new_storage=(storage*)malloc(sizeof(storage));
			new_data=(double*)malloc(total*sizeof(double));
			if (new_storage==NULL || new_data==NULL){
				goto cleanup1;
			}
			
			uint64_t coords[UINT8_MAX];
			for (uint64_t i=0;i<total;i++){
				uint64_t tmp=i;
				for (int16_t i2=(int16_t)out_dim-1;i2>=0;i2--){
					coords[i2]=tmp%new_shape[i2];
					tmp=tmp/new_shape[i2];
				}
				
				uint64_t offset1=in1->offset;
				uint64_t offset2=in2->offset;
				for (uint8_t i3=0;i3<out_dim;i3++){
					offset1+=coords[i3]*new_stride1[i3];
					offset2+=coords[i3]*new_stride2[i3];
				}
				double value1=in1->base->data[offset1];
				double value2=in2->base->data[offset2];
				switch (type) {
				case ADD:
					new_data[i]=value1+value2;
					break;
				case SUB:
					new_data[i]=value1-value2;
					break;
				case MUL:
					new_data[i]=value1*value2;
					break;
				case DIV:
					new_data[i]=value1/value2;
					break;
				default:
					new_data[i]=0;
					break;
				}
			}
			//out初始化
			out->ndim=out_dim;
			out->shape=new_shape;
			out->stride=new_stride;
			out->total_num=total;
			out->offset=0;
			out->base=new_storage;
			new_storage->total_num=total;
			new_storage->refer_count=1;
			new_storage->data=new_data;
			
			back(NDARRAY_OK);
			return;
		}
		
		
		cleanup1:
		free(new_stride);
		free(new_shape);
		free(new_data);
		free(new_storage);
		back(NDARRAY_ERR_ALLOC_FAIL);
		return;
	}
	else{
		//分配
		uint8_t out_dim=in1->ndim>in2->ndim?in1->ndim:in2->ndim;
		if (out->ndim!=out_dim){
			back(NDARRAY_ERR_WRONG_SHAPE);
			return;//形状不匹配
		}
		
		uint64_t out_shape[UINT8_MAX];
		_calculate_shape(in1,in2,out_shape,false);
		for (uint8_t i=0;i<out_dim;i++){
			if (out->shape[i]!=out_shape[i]){
				back(NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
		
		uint64_t new_stride1[UINT8_MAX];
		uint64_t new_stride2[UINT8_MAX];
		_calculate_broadcast(in1,in2,new_stride1,new_stride2,false);
		
		uint64_t total=out->total_num;
		uint64_t coords[UINT8_MAX];
		
		for (uint64_t idx=0;idx<total;idx++){
			uint64_t tmp=idx;
			for (int16_t d=(int16_t)out_dim-1;d>=0;d--){
				coords[d]=tmp%out_shape[d];
				tmp=tmp/out->shape[d];
			}
			
			uint64_t off1=in1->offset;
			uint64_t off2=in2->offset;
			for (uint8_t i=0;i<out_dim;i++){
				off1+=coords[i]*new_stride1[i];
				off2+=coords[i]*new_stride2[i];
			}
			double val1=in1->base->data[off1];
			double val2=in2->base->data[off2];
			
			double res;
			switch (type) {
			case ADD:
				res=val1+val2;
				break;
			case SUB:
				res=val1-val2;
				break;
			case MUL:
				res=val1*val2;
				break;
			case DIV:
				res=val1/val2;
				break;
			default:
				res=0;
				break;
			}
			uint64_t out_off = out->offset;
			for (uint8_t d = 0; d < out_dim; d++) {
				out_off += coords[d] * out->stride[d];
			}
			out->base->data[out_off] = res;
			
		}
		
		back(NDARRAY_OK);
		return;
	}
}

void ndarray_add(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	_calculate_func(in1,in2,out,ADD,back_status);
}

void ndarray_sub(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	_calculate_func(in1,in2,out,SUB,back_status);
}

void ndarray_mul(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	_calculate_func(in1,in2,out,MUL,back_status);
}

void ndarray_div(const ndarray *in1,const ndarray *in2,ndarray *out,uint8_t *back_status){
	_calculate_func(in1,in2,out,DIV,back_status);
}

void ndarray_apply(const ndarray *in,double (*apply_func)(double,void*),void *input,ndarray *out,uint8_t *back_status){
	
}
