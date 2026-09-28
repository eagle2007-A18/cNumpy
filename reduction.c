#include "numpy.h"
#include "internel.h"
#include <math.h>

static inline void _back(uint8_t *back_status, uint8_t kind) {
	if (back_status!=NULL) {
		*back_status=kind;
	}
	return;
}

static inline void _check(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, uint8_t *back_status) {
	if (dim_num>in->ndim || dim_num==0) {
		_back(back_status, NDARRAY_ERR_WRONGDIM);
		return;
	}

	//检查每一个待规约维度有没有超限
	for (uint8_t i=0; i<dim_num; i++) {
		if (dim_list[i]>=in->ndim) {
			_back(back_status, NDARRAY_ERR_DIM_OUT_OF_RANGE);
			return;
		}
	}

	//检查待规约维度有没有重复
	bool visited[UINT8_MAX]= {false};
	for (uint8_t i=0; i<dim_num; i++) {
		uint8_t dim=dim_list[i];
		if (visited[dim]==false) {
			visited[dim]=true;
		}
		else {
			_back(back_status, NDARRAY_ERR_DIM_REPEAT);
			return;
		}
	}

	_back(back_status, NDARRAY_OK);
	return;
}

static void _calculate_out(const ndarray *in,
                           uint8_t dim_num,
                           const uint8_t *dim_list,
                           bool keepdim,
                           bool *is_reduce,
                           uint64_t *out_shape,
                           uint8_t *out_ndim,
                           uint64_t *out_total,
                           uint8_t *back_status) {

	//is_reduce全局置false
	for (uint8_t i=0; i<UINT8_MAX; i++) {
		is_reduce[i]=false;
	}

	for (uint8_t i=0; i<dim_num; i++) {
		is_reduce[dim_list[i]]=true;
	}

	//计算输出维度
	if (keepdim==true) {
		*out_ndim=in->ndim;
	}
	else {
		*out_ndim=in->ndim-dim_num;
	}

	uint8_t out_idx=0;//写入游标
	if (keepdim==true) {
		for (uint8_t i=0; i<in->ndim; i++) {
			if (is_reduce[i]==true) {
				out_shape[i]=1;
			}
			else {
				out_shape[i]=in->shape[i];
			}
		}
	}
	else {
		for (uint8_t i=0; i<in->ndim; i++) {
			if (is_reduce[i]==false) {
				out_shape[out_idx]=in->shape[i];
				out_idx++;
			}
			else {
				//不做任何处理
			}
		}
	}

	//计算out_total
	if (*out_ndim==0) {
		*out_total=1;
	}
	else {
		*out_total=1;
		for (uint8_t i=0; i<*out_ndim; i++) {
			(*out_total)*=out_shape[i];
		}
	}

	_back(back_status, NDARRAY_OK);
	return;
}

void ndarray_sum(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, bool keepdim, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || dim_list==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//dim_num与dim_list检查
	uint8_t back1;
	_check(in, dim_num, dim_list, &back1);

	if (back1!=NDARRAY_OK) {
		_back(back_status, back1);
		return;
	}

	//检查通过，传入合法，开始计算out的形状
	//声明栈上数组
	bool is_reduce[UINT8_MAX];
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim;
	uint64_t out_total;
	_calculate_out(in, dim_num, dim_list, keepdim, is_reduce, out_shape, &out_ndim, &out_total, NULL);

	//大分支，out->base有没有分配
	if (out->base==NULL) {
		//out未分配，创建新存储
		storage *new_storage=(storage*)malloc(sizeof(storage));
		if (new_storage==NULL) {
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//当ndim大于0时，分配shape和stride
		if (out_ndim>0) {
			out->shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				free(new_storage);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape，计算stride
			for (uint8_t i=0; i<out_ndim; i++) {
				out->shape[i]=out_shape[i];
			}
			for (uint8_t i1=0; i1<out_ndim-1; i1++) {
				uint64_t curr=1;
				for (uint8_t i2=i1+1; i2<out_ndim; i2++) {
					curr*=out_shape[i2];
				}
				out->stride[i1]=curr;
			}
			out->stride[out_ndim-1]=1;
		}
		else {
			out->shape=NULL;
			out->stride=NULL;
		}

		//分配data
		new_storage->data=(double*)calloc(out_total, sizeof(double));
		if (new_storage->data==NULL) {
			free(new_storage);
			free(out->shape);
			free(out->stride);
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		out->ndim=out_ndim;
		out->total_num=out_total;
		out->offset=0;
		out->base=new_storage;

		new_storage->refer_count=1;
		new_storage->total_num=out_total;

	}
	else {
		//检查ndim
		if (out->ndim!=out_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		if (out->ndim>0) {
			for (uint8_t i=0; i<out_ndim; i++) {
				if (out->shape[i]!=out_shape[i]) {
					_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
					return;
				}
			}
		}

		//形状校验通过，清空out对应的物理位置
		uint64_t out_coords[UINT8_MAX];
		for (uint64_t idx=0; idx<out_total; idx++) {
			_linear_to_coords(idx, out->ndim, out->shape, out_coords);
			uint64_t offset=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
			out->base->data[offset]=0;
		}
	}

	//累加开始
	uint64_t in_coords[UINT8_MAX];
	uint64_t out_coords[UINT8_MAX];

	for (uint64_t idx=0; idx<in->total_num; idx++) {
		_linear_to_coords(idx, in->ndim, in->shape, in_coords);

		if (keepdim==true) {
			for (uint8_t d = 0; d < in->ndim; d++) {
				if (is_reduce[d] == true) {
					out_coords[d] = 0;
				}
				else {
					out_coords[d] = in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx = 0;
			for (uint8_t d = 0; d < in->ndim; d++) {
				if (is_reduce[d] == false) {
					out_coords[out_idx] = in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_off=_coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_off=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		out->base->data[out_off]+=in->base->data[in_off];
	}
	_back(back_status, NDARRAY_OK);
}

void ndarray_mean(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, bool keepdim, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || dim_list==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//dim_num与dim_list检查
	uint8_t back1;
	_check(in, dim_num, dim_list, &back1);

	if (back1!=NDARRAY_OK) {
		_back(back_status, back1);
		return;
	}

	//检查通过，传入合法，开始计算out的形状
	//声明栈上数组
	bool is_reduce[UINT8_MAX];
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim;
	uint64_t out_total;
	_calculate_out(in, dim_num, dim_list, keepdim, is_reduce, out_shape, &out_ndim, &out_total, NULL);

	//大分支，out->base有没有分配
	if (out->base==NULL) {
		//out未分配，创建新存储
		storage *new_storage=(storage*)malloc(sizeof(storage));
		if (new_storage==NULL) {
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//当ndim大于0时，分配shape和stride
		if (out_ndim>0) {
			out->shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				free(new_storage);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape，计算stride
			for (uint8_t i=0; i<out_ndim; i++) {
				out->shape[i]=out_shape[i];
			}
			for (uint8_t i1=0; i1<out_ndim-1; i1++) {
				uint64_t curr=1;
				for (uint8_t i2=i1+1; i2<out_ndim; i2++) {
					curr*=out_shape[i2];
				}
				out->stride[i1]=curr;
			}
			out->stride[out_ndim-1]=1;
		}
		else {
			out->shape=NULL;
			out->stride=NULL;
		}

		//分配data
		new_storage->data=(double*)calloc(out_total, sizeof(double));
		if (new_storage->data==NULL) {
			free(new_storage);
			free(out->shape);
			free(out->stride);
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		out->ndim=out_ndim;
		out->total_num=out_total;
		out->offset=0;
		out->base=new_storage;

		new_storage->refer_count=1;
		new_storage->total_num=out_total;

	}
	else {
		//检查ndim
		if (out->ndim!=out_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		if (out->ndim>0) {
			for (uint8_t i=0; i<out_ndim; i++) {
				if (out->shape[i]!=out_shape[i]) {
					_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
					return;
				}
			}
		}

		//形状校验通过，清空out对应的物理位置
		uint64_t out_coords[UINT8_MAX];
		for (uint64_t idx=0; idx<out_total; idx++) {
			_linear_to_coords(idx, out->ndim, out->shape, out_coords);
			uint64_t offset=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
			out->base->data[offset]=0;
		}
	}

	//开始求平均
	uint64_t in_coords[UINT8_MAX];
	uint64_t out_coords[UINT8_MAX];

	for (uint64_t i=0; i<in->total_num; i++) {
		_linear_to_coords(i, in->ndim, in->shape, in_coords);
		if (keepdim==true) {
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==true) {
					out_coords[d]=0;
				}
				else {
					out_coords[d]=in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx=0;
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==false) {
					out_coords[out_idx]=in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_linear=_coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_linear=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		out->base->data[out_linear]+=in->base->data[in_linear];
	}

	uint64_t reduce_count = in->total_num / out_total;
	for (uint64_t idx = 0; idx < out_total; idx++) {
		_linear_to_coords(idx, out->ndim, out->shape, out_coords);
		uint64_t off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
		out->base->data[off] /= (double)reduce_count;
	}

	_back(back_status, NDARRAY_OK);
}

void ndarray_variance(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, bool keepdim, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || dim_list==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//dim_num与dim_list检查
	uint8_t back1;
	_check(in, dim_num, dim_list, &back1);

	if (back1!=NDARRAY_OK) {
		_back(back_status, back1);
		return;
	}

	//检查通过，传入合法，开始计算out的形状
	//声明栈上数组
	bool is_reduce[UINT8_MAX];
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim;
	uint64_t out_total;
	_calculate_out(in, dim_num, dim_list, keepdim, is_reduce, out_shape, &out_ndim, &out_total, NULL);

	//大分支，out->base有没有分配
	if (out->base==NULL) {
		//out未分配，创建新存储
		storage *new_storage=(storage*)malloc(sizeof(storage));
		if (new_storage==NULL) {
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//当ndim大于0时，分配shape和stride
		if (out_ndim>0) {
			out->shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				free(new_storage);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape，计算stride
			for (uint8_t i=0; i<out_ndim; i++) {
				out->shape[i]=out_shape[i];
			}
			for (uint8_t i1=0; i1<out_ndim-1; i1++) {
				uint64_t curr=1;
				for (uint8_t i2=i1+1; i2<out_ndim; i2++) {
					curr*=out_shape[i2];
				}
				out->stride[i1]=curr;
			}
			out->stride[out_ndim-1]=1;
		}
		else {
			out->shape=NULL;
			out->stride=NULL;
		}

		//分配data
		new_storage->data=(double*)calloc(out_total, sizeof(double));
		if (new_storage->data==NULL) {
			free(new_storage);
			free(out->shape);
			free(out->stride);
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		out->ndim=out_ndim;
		out->total_num=out_total;
		out->offset=0;
		out->base=new_storage;

		new_storage->refer_count=1;
		new_storage->total_num=out_total;

	}
	else {
		//检查ndim
		if (out->ndim!=out_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		if (out->ndim>0) {
			for (uint8_t i=0; i<out_ndim; i++) {
				if (out->shape[i]!=out_shape[i]) {
					_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
					return;
				}
			}
		}

		//形状校验通过，清空out对应的物理位置
		uint64_t out_coords[UINT8_MAX];
		for (uint64_t idx=0; idx<out_total; idx++) {
			_linear_to_coords(idx, out->ndim, out->shape, out_coords);
			uint64_t offset=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
			out->base->data[offset]=0;
		}
	}

	//第一遍：累加到 out（得到每个输出位置的和）
	uint64_t in_coords[UINT8_MAX];
	uint64_t out_coords[UINT8_MAX];

	for (uint64_t idx=0; idx<in->total_num; idx++) {
		_linear_to_coords(idx, in->ndim, in->shape, in_coords);

		if (keepdim==true) {
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==true) {
					out_coords[d]=0;
				}
				else {
					out_coords[d]=in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx=0;
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==false) {
					out_coords[out_idx]=in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_off  = _coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		out->base->data[out_off] += in->base->data[in_off];
	}

	//计算规约元素总数
	uint64_t reduce_count = in->total_num / out_total;

	//分配临时均值数组
	double *means = (double*)malloc(out_total * sizeof(double));
	if (means == NULL) {
		_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
		return;
	}

	//第一遍后处理：out 里的和变成均值，存入 means，同时把 out 清零给下一遍用
	for (uint64_t idx=0; idx<out_total; idx++) {
		_linear_to_coords(idx, out->ndim, out->shape, out_coords);
		uint64_t off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
		means[idx] = out->base->data[off] / (double)reduce_count;
		out->base->data[off] = 0.0;
	}

	//第二遍：遍历输入，累加 (x - mean)^2
	for (uint64_t idx=0; idx<in->total_num; idx++) {
		_linear_to_coords(idx, in->ndim, in->shape, in_coords);

		if (keepdim==true) {
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==true) {
					out_coords[d]=0;
				}
				else {
					out_coords[d]=in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx=0;
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==false) {
					out_coords[out_idx]=in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_off  = _coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		//计算 out 的逻辑索引，用于查 means
		uint64_t logical = 0;
		for (uint8_t d=0; d<out->ndim; d++) {
			logical = logical * out->shape[d] + out_coords[d];
		}

		double diff = in->base->data[in_off] - means[logical];
		out->base->data[out_off] += diff * diff;
	}

	//第二遍后处理：除以 reduce_count
	for (uint64_t idx=0; idx<out_total; idx++) {
		_linear_to_coords(idx, out->ndim, out->shape, out_coords);
		uint64_t off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
		out->base->data[off] /= (double)reduce_count;
	}

	free(means);

	_back(back_status, NDARRAY_OK);
}

void ndarray_max(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, bool keepdim, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || dim_list==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//dim_num与dim_list检查
	uint8_t back1;
	_check(in, dim_num, dim_list, &back1);

	if (back1!=NDARRAY_OK) {
		_back(back_status, back1);
		return;
	}

	//检查通过，传入合法，开始计算out的形状
	//声明栈上数组
	bool is_reduce[UINT8_MAX];
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim;
	uint64_t out_total;
	_calculate_out(in, dim_num, dim_list, keepdim, is_reduce, out_shape, &out_ndim, &out_total, NULL);

	//大分支，out->base有没有分配
	if (out->base==NULL) {
		//out未分配，创建新存储
		storage *new_storage=(storage*)malloc(sizeof(storage));
		if (new_storage==NULL) {
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//当ndim大于0时，分配shape和stride
		if (out_ndim>0) {
			out->shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				free(new_storage);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape，计算stride
			for (uint8_t i=0; i<out_ndim; i++) {
				out->shape[i]=out_shape[i];
			}
			for (uint8_t i1=0; i1<out_ndim-1; i1++) {
				uint64_t curr=1;
				for (uint8_t i2=i1+1; i2<out_ndim; i2++) {
					curr*=out_shape[i2];
				}
				out->stride[i1]=curr;
			}
			out->stride[out_ndim-1]=1;
		}
		else {
			out->shape=NULL;
			out->stride=NULL;
		}

		//分配data
		new_storage->data=(double*)malloc(out_total*sizeof(double));
		if (new_storage->data==NULL) {
			free(new_storage);
			free(out->shape);
			free(out->stride);
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//把新分配的data置为无穷小
		for (uint64_t i=0; i<out_total; i++) {
			new_storage->data[i]=-INFINITY;
		}

		out->ndim=out_ndim;
		out->total_num=out_total;
		out->offset=0;
		out->base=new_storage;

		new_storage->refer_count=1;
		new_storage->total_num=out_total;

	}
	else {
		//检查ndim
		if (out->ndim!=out_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		if (out->ndim>0) {
			for (uint8_t i=0; i<out_ndim; i++) {
				if (out->shape[i]!=out_shape[i]) {
					_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
					return;
				}
			}
		}

		//形状校验通过，将out对应的物理位置置为无穷小
		uint64_t out_coords[UINT8_MAX];
		for (uint64_t idx=0; idx<out_total; idx++) {
			_linear_to_coords(idx, out->ndim, out->shape, out_coords);
			uint64_t offset=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
			out->base->data[offset]=-INFINITY;
		}
	}

	//开始进行比较
	//遍历输入，比较更新
	uint64_t in_coords[UINT8_MAX];
	uint64_t out_coords[UINT8_MAX];

	for (uint64_t idx=0; idx<in->total_num; idx++) {
		_linear_to_coords(idx, in->ndim, in->shape, in_coords);

		if (keepdim==true) {
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==true) {
					out_coords[d]=0;
				}
				else {
					out_coords[d]=in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx=0;
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==false) {
					out_coords[out_idx]=in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_off  = _coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		double v = in->base->data[in_off];
		if (v > out->base->data[out_off]) {
			out->base->data[out_off] = v;
		}
	}

	_back(back_status, NDARRAY_OK);
}

void ndarray_min(const ndarray *in, uint8_t dim_num, uint8_t *dim_list, bool keepdim, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || dim_list==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//dim_num与dim_list检查
	uint8_t back1;
	_check(in, dim_num, dim_list, &back1);

	if (back1!=NDARRAY_OK) {
		_back(back_status, back1);
		return;
	}

	//检查通过，传入合法，开始计算out的形状
	//声明栈上数组
	bool is_reduce[UINT8_MAX];
	uint64_t out_shape[UINT8_MAX];
	uint8_t out_ndim;
	uint64_t out_total;
	_calculate_out(in, dim_num, dim_list, keepdim, is_reduce, out_shape, &out_ndim, &out_total, NULL);

	//大分支，out->base有没有分配
	if (out->base==NULL) {
		//out未分配，创建新存储
		storage *new_storage=(storage*)malloc(sizeof(storage));
		if (new_storage==NULL) {
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//当ndim大于0时，分配shape和stride
		if (out_ndim>0) {
			out->shape=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(out_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				free(new_storage);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape，计算stride
			for (uint8_t i=0; i<out_ndim; i++) {
				out->shape[i]=out_shape[i];
			}
			for (uint8_t i1=0; i1<out_ndim-1; i1++) {
				uint64_t curr=1;
				for (uint8_t i2=i1+1; i2<out_ndim; i2++) {
					curr*=out_shape[i2];
				}
				out->stride[i1]=curr;
			}
			out->stride[out_ndim-1]=1;
		}
		else {
			out->shape=NULL;
			out->stride=NULL;
		}

		//分配data
		new_storage->data=(double*)malloc(out_total*sizeof(double));
		if (new_storage->data==NULL) {
			free(new_storage);
			free(out->shape);
			free(out->stride);
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		out->ndim=out_ndim;
		out->total_num=out_total;
		out->offset=0;
		out->base=new_storage;

		new_storage->refer_count=1;
		new_storage->total_num=out_total;

		//把新分配的data全部置为无穷大
		for (uint64_t i=0; i<out_total; i++) {
			new_storage->data[i]=INFINITY;
		}

	}
	else {
		//检查ndim
		if (out->ndim!=out_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		if (out->ndim>0) {
			for (uint8_t i=0; i<out_ndim; i++) {
				if (out->shape[i]!=out_shape[i]) {
					_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
					return;
				}
			}
		}

		//形状校验通过，把out对应的物理位置置为无穷大
		uint64_t out_coords[UINT8_MAX];
		for (uint64_t idx=0; idx<out_total; idx++) {
			_linear_to_coords(idx, out->ndim, out->shape, out_coords);
			uint64_t offset=_coords_to_linear(out_coords, out->ndim, out->offset, out->stride);
			out->base->data[offset]=INFINITY;
		}
	}

	//遍历输入，比较更新
	uint64_t in_coords[UINT8_MAX];
	uint64_t out_coords[UINT8_MAX];

	for (uint64_t idx=0; idx<in->total_num; idx++) {
		_linear_to_coords(idx, in->ndim, in->shape, in_coords);

		if (keepdim==true) {
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==true) {
					out_coords[d]=0;
				}
				else {
					out_coords[d]=in_coords[d];
				}
			}
		}
		else {
			uint8_t out_idx=0;
			for (uint8_t d=0; d<in->ndim; d++) {
				if (is_reduce[d]==false) {
					out_coords[out_idx]=in_coords[d];
					out_idx++;
				}
			}
		}

		uint64_t in_off  = _coords_to_linear(in_coords, in->ndim, in->offset, in->stride);
		uint64_t out_off = _coords_to_linear(out_coords, out->ndim, out->offset, out->stride);

		double v = in->base->data[in_off];
		if (v < out->base->data[out_off]) {
			out->base->data[out_off] = v;
		}
	}

	_back(back_status, NDARRAY_OK);
}

