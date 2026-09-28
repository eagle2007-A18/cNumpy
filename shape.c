#include "numpy.h"
#include "internel.h"

static inline void _back(uint8_t *back_status, uint8_t kind) {
	if (back_status!=NULL) {
		*back_status=kind;
	}
	return;
}

void ndarray_reshape(const ndarray *in, uint8_t new_ndim, uint64_t *new_shape, ndarray *out, uint8_t *back_status) {
	//检查传入指针
	if (in==NULL || new_shape==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//检查是否为标量
	if (in->ndim==0) {
		_back(back_status, NDARRAY_ERR_SCALAR_CANNOT_RESHAPE);
		return;
	}

	//检查传入的new_ndim
	if (new_ndim==0) {
		_back(back_status, NDARRAY_ERR_WRONGDIM);
		return;
	}

	//检查shape里面有没有0
	for (uint8_t i=0; i<new_ndim; i++) {
		if (new_shape[i]==0) {
			_back(back_status, NDARRAY_ERR_ZERO_SHAPE);
			return;
		}
	}

	//检查总数
	uint64_t check_total=1;
	for (uint8_t i=0; i<new_ndim; i++) {
		check_total*=new_shape[i];
	}
	if (check_total!=in->total_num) {
		_back(back_status, NDARRAY_ERR_TOTAL_WRONG);
		return;
	}

	//大分支，out有没有分配
	if (out->base==NULL) {
		if (_is_continuous(in)==true) {
			//未分配且内存连续
			out->shape=(uint64_t*)malloc(new_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(new_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape
			for (uint8_t i=0; i<new_ndim; i++) {
				out->shape[i]=new_shape[i];
			}

			//计算stride和total
			_shape_to_stride_total(new_ndim, out->shape, &(out->total_num), out->stride);
			out->base=in->base;
			in->base->refer_count+=1;
			out->offset=in->offset;
			out->ndim=new_ndim;
			_back(back_status, NDARRAY_OK);
			return;
		}
		else {
			//未分配且内存不连续
			out->shape=(uint64_t*)malloc(new_ndim*sizeof(uint64_t));
			out->stride=(uint64_t*)malloc(new_ndim*sizeof(uint64_t));
			if (out->shape==NULL || out->stride==NULL) {
				free(out->shape);
				free(out->stride);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//复制shape
			for (uint8_t i=0; i<new_ndim; i++) {
				out->shape[i]=new_shape[i];
			}

			//计算stride和total
			_shape_to_stride_total(new_ndim, out->shape, &(out->total_num), out->stride);
			out->offset=0;
			out->ndim=new_ndim;

			//分配新storage
			storage *new_storage=(storage*)malloc(sizeof(storage));
			if (new_storage==NULL) {
				free(out->shape);
				free(out->stride);
				out->shape=NULL;
				out->stride=NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}

			//分配new_storage的data
			new_storage->data=(double*)malloc(out->total_num*sizeof(double));
			if (new_storage->data==NULL) {
				free(out->shape);
				free(out->stride);
				out->shape=NULL;
				out->stride=NULL;
				free(new_storage);
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			new_storage->refer_count=1;
			new_storage->total_num=out->total_num;
			out->base=new_storage;

			//按照逻辑位复制data
			uint64_t coord1[UINT8_MAX];
			for (uint64_t i=0; i<in->total_num; i++) {
				_linear_to_coords(i, in->ndim, in->shape, coord1);
				uint64_t in_off=_coords_to_linear(coord1, in->ndim, in->offset, in->stride);
				out->base->data[i]=in->base->data[in_off];
			}
			_back(back_status, NDARRAY_OK);
			return;
		}
	}
	else {
		//out已分配
		//检查ndim
		if (out->ndim!=new_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		for (uint8_t i=0; i<new_ndim; i++) {
			if (out->shape[i]!=new_shape[i]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}

		//检验通过，进行复制
		uint64_t coord1[UINT8_MAX];
		for (uint64_t i=0; i<in->total_num; i++) {
			_linear_to_coords(i, in->ndim, in->shape, coord1);

		}

		uint64_t coord_in[UINT8_MAX];
		uint64_t coord_out[UINT8_MAX];
		for (uint64_t i=0; i<in->total_num; i++) {
			_linear_to_coords(i, in->ndim, in->shape, coord_in);
			_linear_to_coords(i, out->ndim, out->shape, coord_out);
			uint64_t in_off=_coords_to_linear(coord_in, in->ndim, in->offset, in->stride);
			uint64_t out_off=_coords_to_linear(coord_out, out->ndim, out->offset, out->stride);
			out->base->data[out_off]=in->base->data[in_off];
		}

		_back(back_status, NDARRAY_OK);
		return;
	}
}

void ndarray_swapaxes(const ndarray *in, uint8_t dim1, uint8_t dim2, ndarray *out, uint8_t *back_status) {
	//传入指针检查
	if (in==NULL || out==NULL || in->base==NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	//标量检查
	if (in->ndim==0) {
		_back(back_status, NDARRAY_ERR_WRONGDIM);
		return;
	}

	if (dim1==dim2 || dim1>=in->ndim || dim2>=in->ndim) {
		_back(back_status, NDARRAY_ERR_DIM_OUT_OF_RANGE);
		return;
	}

	//在栈上分配new_shape和new_stride
	uint64_t *new_shape=(uint64_t*)alloca(in->ndim*sizeof(uint64_t));
	uint64_t *new_stride=(uint64_t*)alloca(in->ndim*sizeof(uint64_t));

	//复制shape和stride
	for (uint8_t i=0; i<in->ndim; i++) {
		new_shape[i]=in->shape[i];
		new_stride[i]=in->stride[i];
	}

	//交换dim1和dim2
	uint64_t mid;
	mid=new_shape[dim1];
	new_shape[dim1]=new_shape[dim2];
	new_shape[dim2]=mid;

	mid=new_stride[dim1];
	new_stride[dim1]=new_stride[dim2];
	new_stride[dim2]=mid;

	//大分支，是否进行分配
	if (out->base==NULL) {
		//分配new_ndarray的shape和stride
		out->shape=(uint64_t*)malloc(in->ndim*sizeof(uint64_t));
		out->stride=(uint64_t*)malloc(in->ndim*sizeof(uint64_t));
		if (out->shape==NULL || out->stride==NULL) {
			if (out->shape==NULL) {
				free(out->shape);
			}
			if (out->stride==NULL) {
				free(out->stride);
			}
			out->shape=NULL;
			out->stride=NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}

		//复制shape和stride
		for (uint8_t i=0; i<in->ndim; i++) {
			out->shape[i]=new_shape[i];
			out->stride[i]=new_stride[i];
		}

		out->ndim=in->ndim;
		out->total_num=in->total_num;
		out->offset=in->offset;
		out->base=in->base;
		in->base->refer_count+=1;

		_back(back_status, NDARRAY_OK);
		return;
	}
	else {
		//复用内存
		if (out->ndim!=in->ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		//检查shape
		for (uint8_t i=0; i<in->ndim; i++) {
			if (out->shape[i]!=new_shape[i]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}

		uint64_t *coords_out=(uint64_t*)alloca(in->ndim*sizeof(uint64_t));
		uint64_t *coords_in=(uint64_t*)alloca(in->ndim*sizeof(uint64_t));

		for (uint64_t i=0; i<out->total_num; i++) {
			_linear_to_coords(i, out->ndim, out->shape, coords_out);

			for (uint8_t d = 0; d < out->ndim; d++) {
				coords_in[d] = coords_out[d];
			}
			uint64_t swap_tmp = coords_in[dim1];
			coords_in[dim1] = coords_in[dim2];
			coords_in[dim2] = swap_tmp;

			uint64_t in_off  = _coords_to_linear(coords_in,  in->ndim,  in->offset,  in->stride);
			uint64_t out_off = _coords_to_linear(coords_out, out->ndim, out->offset, out->stride);

			out->base->data[out_off] = in->base->data[in_off];
		}
		_back(back_status, NDARRAY_OK);
		return;
	}
}

void ndarray_flatten(const ndarray *in, ndarray *out, uint8_t *back_status) {
	// 传入指针检查
	if (in == NULL || in->base == NULL || out == NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}

	uint64_t total = in->total_num;

	// 大分支，out 是否已分配
	if (out->base == NULL) {
		// 未分配：创建新的一维连续存储（拷贝）
		out->shape = (uint64_t*)malloc(1 * sizeof(uint64_t));
		out->stride = (uint64_t*)malloc(1 * sizeof(uint64_t));
		if (out->shape == NULL || out->stride == NULL) {
			free(out->shape);
			free(out->stride);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		out->shape[0] = total;
		out->stride[0] = 1;
		out->ndim = 1;
		out->total_num = total;
		out->offset = 0;

		// 分配新 storage 与 data
		storage *new_storage = (storage*)malloc(sizeof(storage));
		if (new_storage == NULL) {
			free(out->shape);
			free(out->stride);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		new_storage->data = (double*)malloc(total * sizeof(double));
		if (new_storage->data == NULL) {
			free(out->shape);
			free(out->stride);
			free(new_storage);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		new_storage->refer_count = 1;
		new_storage->total_num = total;
		out->base = new_storage;

		// 按 in 的逻辑 C 顺序复制到 out 连续存储
		uint64_t coord_in[UINT8_MAX];
		for (uint64_t i = 0; i < total; i++) {
			_linear_to_coords(i, in->ndim, in->shape, coord_in);
			uint64_t in_off = _coords_to_linear(coord_in, in->ndim, in->offset, in->stride);
			out->base->data[i] = in->base->data[in_off];
		}

		_back(back_status, NDARRAY_OK);
		return;
	}
	else {
		// 已分配：检查形状，写入已有存储
		if (out->ndim != 1) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		if (out->shape[0] != total) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}

		// 按 in 的逻辑 C 顺序写入 out（out 可能是视图，用 stride 定位）
		uint64_t coord_in[UINT8_MAX];
		for (uint64_t i = 0; i < total; i++) {
			_linear_to_coords(i, in->ndim, in->shape, coord_in);
			uint64_t in_off = _coords_to_linear(coord_in, in->ndim, in->offset, in->stride);
			uint64_t out_off = out->offset + i * out->stride[0];
			out->base->data[out_off] = in->base->data[in_off];
		}

		_back(back_status, NDARRAY_OK);
		return;
	}
}

void ndarray_squeeze(const ndarray *in, ndarray *out, uint8_t dim, bool is_auto, uint8_t *back_status) {
	// 传入指针检查
	if (in == NULL || in->base == NULL || out == NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}
	
	// 标量检查
	if (in->ndim == 0) {
		_back(back_status, NDARRAY_ERR_WRONGDIM);
		return;
	}
	
	// 标记要删除的维度
	bool is_squeeze[UINT8_MAX] = {false};
	if (is_auto == true) {
		for (uint8_t i = 0; i < in->ndim; i++) {
			if (in->shape[i] == 1) {
				is_squeeze[i] = true;
			}
		}
	}
	else {
		if (dim >= in->ndim) {
			_back(back_status, NDARRAY_ERR_DIM_OUT_OF_RANGE);
			return;
		}
		if (in->shape[dim] != 1) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		is_squeeze[dim] = true;
	}
	
	// 构造目标 shape 和 stride
	uint64_t new_shape[UINT8_MAX];
	uint64_t new_stride[UINT8_MAX];
	uint8_t new_ndim = 0;
	for (uint8_t i = 0; i < in->ndim; i++) {
		if (is_squeeze[i] == false) {
			new_shape[new_ndim] = in->shape[i];
			new_stride[new_ndim] = in->stride[i];
			new_ndim++;
		}
	}
	// 计算 new_total，全部删除时为 1
	uint64_t new_total = 1;
	for (uint8_t i = 0; i < new_ndim; i++) {
		new_total *= new_shape[i];
	}
	
	// 大分支，out 是否已分配
	if (out->base == NULL) {
		// 创建视图
		if (new_ndim > 0) {
			out->shape = (uint64_t*)malloc(new_ndim * sizeof(uint64_t));
			out->stride = (uint64_t*)malloc(new_ndim * sizeof(uint64_t));
			if (out->shape == NULL || out->stride == NULL) {
				free(out->shape);
				free(out->stride);
				out->shape = NULL;
				out->stride = NULL;
				_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
				return;
			}
			for (uint8_t i = 0; i < new_ndim; i++) {
				out->shape[i] = new_shape[i];
				out->stride[i] = new_stride[i];
			}
		}
		else {
			out->shape = NULL;
			out->stride = NULL;
		}
		out->ndim = new_ndim;
		out->total_num = new_total;
		out->offset = in->offset;
		out->base = in->base;
		in->base->refer_count += 1;
		
		_back(back_status, NDARRAY_OK);
		return;
	}
	else {
		// 已分配：检查形状
		if (out->ndim != new_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		for (uint8_t i = 0; i < new_ndim; i++) {
			if (out->shape[i] != new_shape[i]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
		
		// 遍历 out 逻辑索引，映射到 in 坐标并复制
		uint64_t coords_out[UINT8_MAX];
		uint64_t coords_in[UINT8_MAX];
		for (uint64_t i = 0; i < new_total; i++) {
			_linear_to_coords(i, out->ndim, out->shape, coords_out);
			
			// 把 out 坐标映射回 in 坐标：被删除的维度坐标填 0
			uint8_t out_idx = 0;
			for (uint8_t d = 0; d < in->ndim; d++) {
				if (is_squeeze[d] == true) {
					coords_in[d] = 0;
				}
				else {
					coords_in[d] = coords_out[out_idx];
					out_idx++;
				}
			}
			
			uint64_t in_off  = _coords_to_linear(coords_in,  in->ndim,  in->offset,  in->stride);
			uint64_t out_off = _coords_to_linear(coords_out, out->ndim, out->offset, out->stride);
			out->base->data[out_off] = in->base->data[in_off];
		}
		
		_back(back_status, NDARRAY_OK);
		return;
	}
}

void ndarray_unsqueeze(const ndarray *in, ndarray *out, uint8_t dim, uint8_t *back_status) {
	// 传入指针检查
	if (in == NULL || in->base == NULL || out == NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}
	
	// 维度检查：允许 dim == in->ndim（插到末尾）
	if (dim > in->ndim) {
		_back(back_status, NDARRAY_ERR_DIM_OUT_OF_RANGE);
		return;
	}
	
	// 构造目标 shape 和 stride
	uint8_t new_ndim = in->ndim + 1;
	uint64_t new_shape[UINT8_MAX];
	uint64_t new_stride[UINT8_MAX];
	
	for (uint8_t i = 0; i < dim; i++) {
		new_shape[i] = in->shape[i];
		new_stride[i] = in->stride[i];
	}
	new_shape[dim] = 1;
	new_stride[dim] = 1;
	for (uint8_t i = dim; i < in->ndim; i++) {
		new_shape[i + 1] = in->shape[i];
		new_stride[i + 1] = in->stride[i];
	}
	
	uint64_t new_total = in->total_num; // 插入长度 1 的维度不改变元素数
	
	// 大分支，out 是否已分配
	if (out->base == NULL) {
		// 创建视图
		out->shape = (uint64_t*)malloc(new_ndim * sizeof(uint64_t));
		out->stride = (uint64_t*)malloc(new_ndim * sizeof(uint64_t));
		if (out->shape == NULL || out->stride == NULL) {
			free(out->shape);
			free(out->stride);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		for (uint8_t i = 0; i < new_ndim; i++) {
			out->shape[i] = new_shape[i];
			out->stride[i] = new_stride[i];
		}
		out->ndim = new_ndim;
		out->total_num = new_total;
		out->offset = in->offset;
		out->base = in->base;
		in->base->refer_count += 1;
		
		_back(back_status, NDARRAY_OK);
		return;
	}
	else {
		// 已分配：检查形状
		if (out->ndim != new_ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		for (uint8_t i = 0; i < new_ndim; i++) {
			if (out->shape[i] != new_shape[i]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
		
		// 遍历 out 逻辑索引，去掉插入维度后映射到 in 坐标
		uint64_t coords_out[UINT8_MAX];
		uint64_t coords_in[UINT8_MAX];
		for (uint64_t i = 0; i < new_total; i++) {
			_linear_to_coords(i, out->ndim, out->shape, coords_out);
			
			// 跳过插入的维度 dim
			uint8_t in_idx = 0;
			for (uint8_t d = 0; d < new_ndim; d++) {
				if (d == dim) {
					continue;
				}
				coords_in[in_idx] = coords_out[d];
				in_idx++;
			}
			
			uint64_t in_off  = _coords_to_linear(coords_in,  in->ndim,  in->offset,  in->stride);
			uint64_t out_off = _coords_to_linear(coords_out, out->ndim, out->offset, out->stride);
			out->base->data[out_off] = in->base->data[in_off];
		}
		
		_back(back_status, NDARRAY_OK);
		return;
	}
}

void ndarray_cat(const ndarray *in_list, uint64_t cat_number, uint8_t cat_dim, ndarray *out, uint8_t *back_status) {
	// 传入指针检查
	if (in_list == NULL || out == NULL) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}
	if (cat_number == 0) {
		_back(back_status, NDARRAY_ERR_NULLPTR);
		return;
	}
	
	// 检查每个输入
	uint8_t ndim = in_list[0].ndim;
	for (uint64_t i = 0; i < cat_number; i++) {
		if (in_list[i].base == NULL) {
			_back(back_status, NDARRAY_ERR_NULLPTR);
			return;
		}
		if (in_list[i].ndim != ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
	}
	
	// 检查 cat_dim
	if (cat_dim >= ndim) {
		_back(back_status, NDARRAY_ERR_DIM_OUT_OF_RANGE);
		return;
	}
	
	// 计算目标形状
	uint64_t out_shape[UINT8_MAX];
	for (uint8_t d = 0; d < ndim; d++) {
		out_shape[d] = in_list[0].shape[d];
	}
	out_shape[cat_dim] = 0;
	for (uint64_t i = 0; i < cat_number; i++) {
		for (uint8_t d = 0; d < ndim; d++) {
			if (d != cat_dim && in_list[i].shape[d] != out_shape[d]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
		out_shape[cat_dim] += in_list[i].shape[cat_dim];
	}
	
	// 计算总元素数
	uint64_t out_total = 1;
	for (uint8_t d = 0; d < ndim; d++) {
		out_total *= out_shape[d];
	}
	
	// 大分支，out 是否已分配
	if (out->base == NULL) {
		// 分配新存储
		out->shape = (uint64_t*)malloc(ndim * sizeof(uint64_t));
		out->stride = (uint64_t*)malloc(ndim * sizeof(uint64_t));
		if (out->shape == NULL || out->stride == NULL) {
			free(out->shape);
			free(out->stride);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		for (uint8_t d = 0; d < ndim; d++) {
			out->shape[d] = out_shape[d];
		}
		_shape_to_stride_total(ndim, out->shape, &out->total_num, out->stride);
		out->ndim = ndim;
		out->offset = 0;
		
		storage *new_storage = (storage*)malloc(sizeof(storage));
		if (new_storage == NULL) {
			free(out->shape);
			free(out->stride);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		new_storage->data = (double*)malloc(out_total * sizeof(double));
		if (new_storage->data == NULL) {
			free(out->shape);
			free(out->stride);
			free(new_storage);
			out->shape = NULL;
			out->stride = NULL;
			_back(back_status, NDARRAY_ERR_ALLOC_FAIL);
			return;
		}
		new_storage->refer_count = 1;
		new_storage->total_num = out_total;
		out->base = new_storage;
	}
	else {
		// 已分配：检查形状
		if (out->ndim != ndim) {
			_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
			return;
		}
		for (uint8_t d = 0; d < ndim; d++) {
			if (out->shape[d] != out_shape[d]) {
				_back(back_status, NDARRAY_ERR_WRONG_SHAPE);
				return;
			}
		}
	}
	
	// 按输入顺序复制数据
	uint64_t cat_offset = 0;
	uint64_t coords_in[UINT8_MAX];
	uint64_t coords_out[UINT8_MAX];
	
	for (uint64_t i = 0; i < cat_number; i++) {
		const ndarray *src = &in_list[i];
		for (uint64_t idx = 0; idx < src->total_num; idx++) {
			_linear_to_coords(idx, src->ndim, src->shape, coords_in);
			
			// 构造 out 坐标：cat_dim 处加上偏移，其余维度保持一致
			for (uint8_t d = 0; d < ndim; d++) {
				if (d == cat_dim) {
					coords_out[d] = coords_in[d] + cat_offset;
				}
				else {
					coords_out[d] = coords_in[d];
				}
			}
			
			uint64_t in_off  = _coords_to_linear(coords_in,  src->ndim, src->offset,  src->stride);
			uint64_t out_off = _coords_to_linear(coords_out, out->ndim, out->offset, out->stride);
			out->base->data[out_off] = src->base->data[in_off];
		}
		cat_offset += src->shape[cat_dim];
	}
	
	_back(back_status, NDARRAY_OK);
}
