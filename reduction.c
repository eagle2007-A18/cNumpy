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
	//传入指针检查
	if (in==NULL || in->base==NULL || dim_list==NULL || out==NULL){
		_back(back_status,NDARRAY_ERR_NULLPTR);
		return;
	}
	
	uint8_t back1;
	//dim_num和dim_list检查
	_check(in,dim_num,dim_list,&back1);
	if (back1!=NDARRAY_OK){
		_back(back_status,back1);
		return;
	}
	
	//辅助标记数组
	bool is_reduce[UINT8_MAX]={false};
	for (uint8_t i=0;i<dim_num;i++){
		is_reduce[dim_list[i]]=true;
	}
	
	//收集规约轴和保留轴
	uint8_t reduce_dims[UINT8_MAX];
	uint8_t reduce_cnt=0;
	uint8_t keep_dims[UINT8_MAX];
	uint8_t keep_cnt=0;
	for (uint8_t d=0;d<in->ndim;d++){
		if (is_reduce[d]){
			reduce_dims[reduce_cnt]=d;
			reduce_cnt++;
		}
		else{
			keep_dims[keep_cnt]=d;
			keep_cnt++;
		}
	}
	
	//计算输出形状
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim=0;
	for (uint8_t d=0;d<in->ndim;d++){
		if (is_reduce[d]==true){
			if (keepdim==true){
				out_shape[out_ndim]=1;
				out_ndim+=1;
			}
			else{
				out_shape[out_ndim]=in->shape[d];
				out_ndim+=1;
			}
		}
	}
	
	//大分支，判断out是否分配
	if (out->base==NULL){
		if (out_ndim==0){
			storage *new_storage=(storage*)malloc(sizeof(storage));
			double *data=(double*)malloc(sizeof(double));
			if (new_storage==NULL || data==NULL){
				free(new_storage);
				free(data);
				_back(back_status,NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			new_storage->data=data;
			new_storage->refer_count=1;
			new_storage->total_num=1;
			out->ndim=0;
			out->shape=NULL;
			out->stride=NULL;
			out->offset=0;
			out->total_num=1;
			out->base=new_storage;
		}
		else{
			uint64_t *new_shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			uint64_t *new_stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (new_shape==NULL || new_stride==NULL){
				free(new_shape);
				free(new_stride);
				_back(back_status,NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			for (uint8_t i=0;i<out_ndim;i++){
				new_shape[i]=out_shape[i];
			}
			uint64_t total;
			_shape_to_stride_total(out_ndim,new_shape,&total,new_stride);
			
			storage *new_storage=(storage*)malloc(sizeof(storage));
			double *data=(double*)malloc(total*sizeof(double));
			if (new_storage==NULL || data==NULL){
				free(new_storage);
				free(data);
				free(new_shape);
				free(new_stride);
				_back(back_status,NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			new_storage->data=data;
			new_storage->refer_count=1;
			new_storage->total_num=total;
			out->ndim=out_ndim;
			out->shape=new_shape;
			out->stride=new_stride;
			out->total_num=total;
			out->offset=0;
			out->base=new_storage;
		}
	}
	else{
		if (out->ndim!=out_ndim){
			_back(back_status,NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		for (uint8_t i=0;i<out_ndim;i++){
			if (out->shape[i]!=out_shape[i]){
				_back(back_status,NDARRAY_ERR_WRONG_SHAPE);
			}
		}
	}
	uint64_t reduce_shape[UINT8_MAX];
	for (uint8_t i=0;i<reduce_cnt;i++){
		reduce_shape[i]=in->shape[reduce_dims[i]];
	}
	uint64_t reduce_total=1;
	for (uint8_t i=0;i<reduce_cnt;i++){
		reduce_total*=reduce_shape[i];
	}
	
	int8_t out_pos_for_in_dim[UINT8_MAX];
	uint8_t out_pos=0;
	for (uint8_t d=0;d<in->ndim;d++){
		if (is_reduce[d]==true){
			out_pos_for_in_dim[d]=-1;
		}
		else{
			out_pos_for_in_dim[d]=out_pos;
			out_pos+=1;
		}
	}
	
	uint8_t reduce_idx_for_in_dim[UINT8_MAX];
	for (uint8_t i=0;i<reduce_cnt;i++){
		reduce_idx_for_in_dim[reduce_dims[i]] = i;
	}
	
	//核心计算
	uint64_t coords_out[UINT8_MAX];
	uint64_t coords_in[UINT8_MAX];
	uint64_t reduce_coords[UINT8_MAX];
	uint64_t total_out=out->total_num;
	
	for (uint64_t idx_out=0;idx_out<total_out;idx_out++){
		if (out_ndim>0){
			_linear_to_coords(idx_out,out_ndim,out->shape,coords_out);
		}
		
		double sum=0.0;
		for (uint64_t idx_red=0;idx_red<reduce_total;idx_red++){
			if (reduce_cnt>0){
				_linear_to_coords(idx_red,reduce_cnt,reduce_shape,reduce_coords);
			}
		}
	}
}

void ndarray_mean(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_variance(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_max(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

void ndarray_min(const ndarray *in,uint8_t dim_num,uint8_t *dim_list,bool keepdim,ndarray *out,uint8_t *back_status){
	
}

