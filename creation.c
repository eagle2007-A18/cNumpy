#include "numpy.h"
#include "internel.h"
#include <math.h>

#define M_PI		3.14159265358979323846

static double _uniform(double start,double end){
	double u=(double)rand()/((double)RAND_MAX+1.0);
	return start+(end-start)*u;
}

static double _nature(double mean,double varience){
	static uint8_t has_space=0;
	static double space;
	if (has_space==1){
		has_space=0;
		return mean+sqrt(varience)*space;
	}
	double u1,u2;
	do{
		u1=(double)rand()/RAND_MAX;
		u2=(double)rand()/RAND_MAX;
	}while(u1==0.0 || u2==0.0);
	
	double z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	double z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
	
	space = z2;
	has_space = 1;
	return mean + sqrt(varience) * z1;
}

static ndarray*_ndarray_init(uint8_t ndim,uint64_t *shape,uint8_t *back_status){
	if (ndim>0 && shape==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return NULL;
	}
	
	//需要使用的内存
	ndarray *new_ndarray=NULL;
	storage *new_storage=NULL;
	uint64_t *new_shape=NULL;
	uint64_t *new_stride=NULL;
	double *data=NULL;
	
	new_ndarray=(ndarray*)malloc(sizeof(ndarray));
	new_storage=(storage*)malloc(sizeof(storage));
	if (new_ndarray==NULL || new_storage==NULL){
		goto cleanup1;
	}
	
	if (ndim==0){//标量
		new_ndarray->base=new_storage;
		data=(double*)malloc(sizeof(double));
		if (data==NULL){
			goto cleanup1;
		}
		
		new_storage->data=data;
		new_storage->refer_count=1;
		new_storage->total_num=1;
		
		new_ndarray->ndim=ndim;
		new_ndarray->shape=new_shape;
		new_ndarray->stride=new_stride;
		new_ndarray->offset=0;
		new_ndarray->total_num=1;
		
		if (back_status!=NULL){
			*back_status=NDARRAY_OK;
		}
		return new_ndarray;
	}
	else{//非标量
		for (uint8_t i=0;i<ndim;i++){
			if (shape[i]==0){
				goto cleanup2;
			}
		}
		
		new_shape=(uint64_t*)malloc(ndim*sizeof(uint64_t));
		new_stride=(uint64_t*)malloc(ndim*sizeof(uint64_t));
		if (new_shape==NULL || new_stride==NULL){
			goto cleanup1;
		}
		
		//复制shape,计算stride和total
		uint64_t total=1;
		for (uint8_t i=0;i<ndim;i++){
			new_shape[i]=shape[i];
			total*=shape[i];
		}
		for (uint8_t i1=0;i1<ndim-1;i1++){
			uint64_t curr=1;
			for (uint8_t i2=i1+1;i2<ndim;i2++){
				curr*=shape[i2];
			}
			new_stride[i1]=curr;
		}
		new_stride[ndim-1]=1;
		
		//分配data
		data=(double*)malloc(total*sizeof(double));
		if (data==NULL){
			goto cleanup1;
		}
		
		//挂载
		new_ndarray->base=new_storage;
		new_ndarray->ndim=ndim;
		new_ndarray->shape=new_shape;
		new_ndarray->stride=new_stride;
		new_ndarray->total_num=total;
		new_ndarray->offset=0;
		
		new_storage->refer_count=1;
		new_storage->total_num=total;
		new_storage->data=data;
		
		if (back_status!=NULL){
			*back_status=NDARRAY_OK;
		}
		return new_ndarray;
	}
	cleanup1:
	free(new_storage);
	free(new_ndarray);
	free(new_shape);
	free(new_stride);
	free(data);
	if (back_status!=NULL){
		*back_status=NDARRAY_ERR_ALLOC_FAIL;
	}
	return NULL;
	
	cleanup2:
	free(new_storage);
	free(new_ndarray);
	free(new_shape);
	free(new_stride);
	free(data);
	if (back_status!=NULL){
		*back_status=NDARRAY_ERR_ZERO_SHAPE;
	}
	return NULL;
}

ndarray* ndarray_init_empty(uint8_t *back_status){
	ndarray *new_ndarray=(ndarray*)malloc(sizeof(ndarray));
	if (new_ndarray==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_ALLOC_FAIL;
		}
		return NULL;
	}
	new_ndarray->ndim=0;
	new_ndarray->shape=NULL;
	new_ndarray->stride=NULL;
	new_ndarray->offset=0;
	new_ndarray->base=NULL;
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return new_ndarray;
}

ndarray* ndarray_init_full(uint8_t ndim,uint64_t *shape,double full_num,uint8_t *back_status){
	uint8_t back;
	ndarray *new_ndarray=_ndarray_init(ndim,shape,&back);
	if (back!=NDARRAY_OK){
		if (back_status!=NULL){
			*back_status=back;
		}
		return NULL;
	}
	
	for (uint64_t i=0;i<new_ndarray->total_num;i++){
		new_ndarray->base->data[i]=full_num;
	}
	
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return new_ndarray;
}

ndarray* ndarray_init_random_uniform(uint8_t ndim,uint64_t *shape,double start,double end,uint8_t *back_status){
	uint8_t back;
	ndarray *new_ndarray=_ndarray_init(ndim,shape,&back);
	if (back!=NDARRAY_OK){
		if (back_status!=NULL){
			*back_status=back;
		}
		return NULL;
	}
	
	for (uint8_t i=0;i<new_ndarray->total_num;i++){
		new_ndarray->base->data[i]=_uniform(start,end);
	}
	
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return new_ndarray;
}

ndarray* ndarray_init_random_nature(uint8_t ndim,uint64_t *shape,double mean,double variance,uint8_t *back_status){
	uint8_t back;
	ndarray *new_ndarray=_ndarray_init(ndim,shape,&back);
	if (back!=NDARRAY_OK){
		if (back_status!=NULL){
			*back_status=back;
		}
		return NULL;
	}
	
	for (uint8_t i=0;i<new_ndarray->total_num;i++){
		new_ndarray->base->data[i]=_nature(mean,variance);
	}
	
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return new_ndarray;
}

void ndarray_free(ndarray *in){
	if (in->base==NULL){
		free(in);
		return;
	}
	else{
		in->base->refer_count-=1;
		if (in->base->refer_count==0){
			free(in->base->data);
			free(in->base);
		}
		
		if (in->ndim==0){
			free(in);
			return;
		}
		else{
			free(in->shape);
			free(in->stride);
			free(in);
			return;
		}
	}
}

void ndarray_empty(ndarray *in){
	if (in->base==NULL){
		in->ndim = 0;
		in->shape = NULL;
		in->stride = NULL;
		in->offset = 0;
		in->total_num = 0;
		in->base = NULL;
		return;
	}
	
	in->base->refer_count-=1;
	if (in->base->refer_count==0){
		free(in->base->data);
		free(in->base);
	}
	
	if (in->ndim!=0){
		free(in->shape);
		free(in->stride);
	}
	
	in->ndim=0;
	in->shape=NULL;
	in->stride=NULL;
	in->base=NULL;
	in->total_num=0;
	in->offset=0;
}

ndarray* ndarray_copy(const ndarray *from,uint8_t *back_status){
	if (from==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_NULLPTR;
		}
		return NULL;
	}
	
	if (from->base==NULL){
		uint8_t back;
		ndarray *empty_ndarray=ndarray_init_empty(&back);
		if (back_status!=NULL){
			*back_status=back;
		}
		return empty_ndarray;
	}
	
	//需要用到的指针
	ndarray *new_ndarray=NULL;
	storage *new_storage=NULL;
	uint64_t *new_shape=NULL;
	uint64_t *new_stride=NULL;
	double *data=NULL;
	
	//分配ndarray结构
	uint8_t back1;
	new_ndarray=ndarray_init_empty(&back1);
	if (back1==NDARRAY_ERR_ALLOC_FAIL){
		goto cleanup1;
	}
	
	//分配storage
	new_storage=(storage*)malloc(sizeof(storage));
	if (new_storage==NULL){
		goto cleanup1;
	}
	
	//分配shape和stride
	if (from->ndim!=0){
		new_shape=(uint64_t*)malloc(from->ndim*sizeof(uint64_t));
		new_stride=(uint64_t*)malloc(from->ndim*sizeof(uint64_t));
		if (new_shape==NULL || new_stride==NULL){
			goto cleanup1;
		}
	}
	
	//分配data
	data=(double*)malloc(from->total_num*sizeof(double));
	if (data==NULL){
		goto cleanup1;
	}
	
	if (from->ndim==0){
		//标量
		new_storage->data=data;
		new_storage->refer_count=1;
		new_storage->total_num=1;
		new_ndarray->base=new_storage;
		new_ndarray->ndim=0;
		new_ndarray->offset=0;
		new_ndarray->shape=NULL;
		new_ndarray->stride=NULL;
		new_storage->data[0]=from->base->data[0];
		if (back_status!=NULL){
			*back_status=NDARRAY_OK;
		}
		return new_ndarray;
	}
	else{
		//非标量
		//复制shape
		for (uint8_t i=0;i<from->ndim;i++){
			new_shape[i]=from->shape[i];
		}
		
		//计算连续存储下的stride
		for (uint8_t i1=0;i1<from->ndim-1;i1++){
			uint64_t curr=1;
			for (uint8_t i2=i1+1;i2<from->ndim;i2++){
				curr*=from->shape[i2];
			}
			new_stride[i1]=curr;
		}
		new_stride[from->ndim-1]=1;
		
		//原始数据复制
		bool is_c=_is_continuous(from);
		if (is_c==true){
			//连续
			uint64_t src_offset=from->offset;
			for (uint64_t i=0;i<from->total_num;i++){
				new_storage->data[i]=from->base->data[src_offset+i];
			}
		}
		else{
			//不连续
			uint64_t flat_index[UINT8_MAX];
			for (uint64_t idx=0;idx<from->total_num;idx++){
				uint64_t temp=idx;
				for (int16_t d=(int16_t)from->ndim-1;d>=0;d--){
					flat_index[d]=temp%from->shape[d];
					temp/=from->shape[d];
				}
				uint64_t phys_offset = from->offset;
				for (uint8_t d = 0; d < from->ndim; d++) {
					phys_offset += flat_index[d] * from->stride[d];
				}
				data[idx] = from->base->data[phys_offset];
			}
		}
		new_storage->data=data;
		new_storage->refer_count=1;
		new_storage->total_num=from->total_num;
		
		new_ndarray->base=new_storage;
		new_ndarray->ndim=from->ndim;
		new_ndarray->shape=new_shape;
		new_ndarray->stride=new_stride;
		new_ndarray->offset=0;
		new_ndarray->total_num=from->total_num;
		if (back_status!=NULL){
			*back_status=NDARRAY_OK;
		}
		return new_ndarray;
	}
	
	cleanup1:
	free(new_ndarray);
	free(new_storage);
	free(new_shape);
	free(new_stride);
	free(data);
	if (back_status!=NULL){
		*back_status=NDARRAY_ERR_ALLOC_FAIL;
	}
	return NULL;
}

ndarray* ndarray_view(const ndarray *father,uint8_t *back_status){
	if (father->base==NULL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_EMPTY_VIEW;
		}
		return NULL;//空ndarray不可拥有视图
	}
	
	uint8_t back;
	ndarray *new_ndarray=ndarray_init_empty(&back);
	if (back==NDARRAY_ERR_ALLOC_FAIL){
		if (back_status!=NULL){
			*back_status=NDARRAY_ERR_ALLOC_FAIL;
		}
		return NULL;
	}
	if (father->ndim!=0){
		new_ndarray->shape=(uint64_t*)malloc(father->ndim*sizeof(uint64_t));
		new_ndarray->stride=(uint64_t*)malloc(father->ndim*sizeof(uint64_t));
		if (new_ndarray->shape==NULL || new_ndarray->stride==NULL){
			if (new_ndarray->shape==NULL){
				free(new_ndarray->stride);
			}
			else{
				free(new_ndarray->shape);
			}
			free(new_ndarray);
			if (back_status!=NULL){
				*back_status=NDARRAY_ERR_ALLOC_FAIL;
			}
			return NULL;
		}
	}
	
	new_ndarray->ndim=father->ndim;
	new_ndarray->offset=father->offset;
	new_ndarray->total_num=father->total_num;
	
	for (uint8_t i=0;i<father->ndim;i++){
		new_ndarray->shape[i]=father->shape[i];
		new_ndarray->stride[i]=father->stride[i];
	}
	
	new_ndarray->base=father->base;
	new_ndarray->base->refer_count+=1;
	if (back_status!=NULL){
		*back_status=NDARRAY_OK;
	}
	return new_ndarray;
}
