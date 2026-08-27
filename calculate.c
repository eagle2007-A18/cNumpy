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

static inline double _cal(double num1,double num2,uint8_t kind){
	double ans;
	switch (kind) {
	case ADD:
		ans=num1+num2;
		break;
	case SUB:
		ans=num1-num2;
		break;
	case MUL:
		ans=num1*num2;
		break;
	case DIV:
		ans=num1/num2;
		break;
	default:
		ans=0;
		break;
	}
	return ans;
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
			new_data[0]=_cal(in1->base->data[in1->offset],in2->base->data[in2->offset],type);
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
				_linear_to_coords(i,out_dim,new_shape,coords);
				
				uint64_t offset1=_coords_to_linear(coords,out_dim,in1->offset,new_stride1);
				uint64_t offset2=_coords_to_linear(coords,out_dim,in2->offset,new_stride2);
				
				double value1=in1->base->data[offset1];
				double value2=in2->base->data[offset2];
				new_data[i]=_cal(value1,value2,type);
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
			_linear_to_coords(idx,out_dim,out_shape,coords);
			
			uint64_t off1=_coords_to_linear(coords,out_dim,in1->offset,new_stride1);
			uint64_t off2=_coords_to_linear(coords,out_dim,in2->offset,new_stride2);
			double val1=in1->base->data[off1];
			double val2=in2->base->data[off2];
			
			double res;
			res=_cal(val1,val2,type);
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

void ndarray_apply(const ndarray *in, double (*apply_func)(double,void*), void *input, ndarray *out, uint8_t *back_status){
	// 传入指针检查
	if (in == NULL || out == NULL || in->base == NULL || apply_func == NULL) {
		back(NDARRAY_ERR_NULLPTR);
		return;
	}
	
	uint8_t ndim = in->ndim;
	uint64_t total = in->total_num;
	
	// out 未分配：创建新存储
	if (out->base == NULL) {
		// 分配 shape 和 stride（连续内存）
		uint64_t *new_shape = NULL;
		uint64_t *new_stride = NULL;
		if (ndim > 0) {
			new_shape = (uint64_t*)malloc(ndim * sizeof(uint64_t));
			new_stride = (uint64_t*)malloc(ndim * sizeof(uint64_t));
			if (new_shape == NULL || new_stride == NULL) {
				free(new_shape); free(new_stride);
				back(NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			for (uint8_t i = 0; i < ndim; i++) new_shape[i] = in->shape[i];
			for (uint8_t i = 0; i < ndim - 1; i++) {
				uint64_t curr = 1;
				for (uint8_t j = i + 1; j < ndim; j++) curr *= new_shape[j];
				new_stride[i] = curr;
			}
			new_stride[ndim - 1] = 1;
		}
		
		double *new_data = (double*)malloc(total * sizeof(double));
		storage *new_storage = (storage*)malloc(sizeof(storage));
		if (new_data == NULL || new_storage == NULL) {
			free(new_shape); free(new_stride); free(new_data); free(new_storage);
			back(NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		
		uint64_t coords[UINT8_MAX];
		for (uint64_t idx = 0; idx < total; idx++) {
			_linear_to_coords(idx, ndim, in->shape, coords);
			uint64_t in_off = _coords_to_linear(coords, ndim, in->offset, in->stride);
			new_data[idx] = apply_func(in->base->data[in_off], input);
		}
		
		// 初始化 out
		out->ndim = ndim;
		out->shape = new_shape;
		out->stride = new_stride;
		out->total_num = total;
		out->offset = 0;
		out->base = new_storage;
		new_storage->data = new_data;
		new_storage->refer_count = 1;
		new_storage->total_num = total;
		
		back(NDARRAY_OK);
		return;
	}
	
	// out 已分配：检查形状匹配
	if (in->ndim != out->ndim) {
		back(NDARRAY_ERR_WRONG_SHAPE);
		return;
	}
	for (uint8_t i = 0; i < ndim; i++) {
		if (in->shape[i] != out->shape[i]) {
			back(NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
	}
	
	// 遍历并写入 out（可能是视图）
	uint64_t coords[UINT8_MAX];
	for (uint64_t idx = 0; idx < total; idx++) {
		_linear_to_coords(idx, ndim, in->shape, coords);
		uint64_t in_off  = _coords_to_linear(coords, ndim, in->offset,  in->stride);
		uint64_t out_off = _coords_to_linear(coords, ndim, out->offset, out->stride);
		out->base->data[out_off] = apply_func(in->base->data[in_off], input);
	}
	
	back(NDARRAY_OK);
}
