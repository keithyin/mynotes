# 看 roi 源码学习 tensorflow 如何自定义 op



## register 看起

```c++
REGISTER_OP("RoiPool")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax: int32");
// Attr 属性，一旦注册到图上的时候定了，就不能再修改了哦。

REGISTER_OP("RoiPoolGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax: int32")
    .Input("grad: T")
    .Output("output: T");
```



### Op 实现部分

* OP_REQUIRES_OK（判断函数） 与 OP_REQUIRES

```c++
template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }
  //构造函数的代码是在初始化图的时候执行的，只执行一次，sess.run 时候执行的是 Compute部分代码！！！
  

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor，0,1代表 第几个的输入
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    
    //将输入Tensor 给展平，目的何在？？
    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();

    // data should have 4 dimensions.运行时dim检查
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // ROIs 的数量
    int num_rois = bottom_rois.dim_size(0);
    // mini-batch 的大小
    int batch_size = bottom_data.dim_size(0);
    // 数据的 高 row
    int data_height = bottom_data.dim_size(1);
    // 数据的宽
    int data_width = bottom_data.dim_size(2);
    // 输入 channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    
    //MakeShape 用来设置 TensorShape。
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // 创建输出Tensor，然后用context给其分配 空间。
    // OP_REQUIRES_OK 用来检查分配空间时是否出了问题
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    
    //为啥操作的时候非要展平
    auto output = output_tensor->template flat<T>();

    Tensor* argmax_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    
    //又展平了
    auto argmax = argmax_tensor->template flat<int>();

    int pooled_height = pooled_height_;
    int pooled_width = pooled_width_;
    float spatial_scale = spatial_scale_;

    auto shard = [pooled_height, pooled_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &output, &argmax]
                  (int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, ph, pw, c) 是roi pooled 输出的一个元素。
        int n = b;
        int c = n % num_channels; //num_channels ，输入channels个数
        n /= num_channels;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;
        
        // 这一串代码，就是为了确定 b 所在的 tensor 的位置。b是一个一维索引，
        // 需要处理一下再转成 四维
        /*********************************************************/
		
        // 找到第 n 个 样本的 roi 起始位置。
        const float* bottom_rois = bottom_rois_flat.data() + n * 5;
        int roi_batch_ind = bottom_rois[0];
        
        // spatial_scale 是 roi 被 reshape 时的 scale。
        // 原始图片 reshape 后，才用来训练的，所以 reshape 后的
        // roi 也得跟着改变一下，这样保证roi 还是 roi。
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        
        // 这个部分是用来求 当前 roi 应该怎么 pool 的代码，
        // 计算了 max_pool 的 kernel 大小。
        const T bin_size_h = static_cast<T>(roi_height)
                           / static_cast<T>(pooled_height);
        const T bin_size_w = static_cast<T>(roi_width)
                           / static_cast<T>(pooled_width);
		
        //计算了 roi 的内部偏移
        int hstart = static_cast<int>(floor(ph * bin_size_h));
        int wstart = static_cast<int>(floor(pw * bin_size_w));
        int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));
		
        // 将roi内部偏移加到 roi 相对原始图片的位置，就得到了roi 在原图中的绝对位置
        hstart = std::min(std::max(hstart + roi_start_h, 0), data_height);
        hend = std::min(std::max(hend + roi_start_h, 0), data_height);
        wstart = std::min(std::max(wstart + roi_start_w, 0), data_width);
        wend = std::min(std::max(wend + roi_start_w, 0), data_width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        const float* bottom_data = bottom_data_flat.data() + roi_batch_ind * num_channels * data_height * data_width;
        
        //可以看出，一次就求一个 roi 的一个位置的  max pool。
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = (h * data_width + w) * num_channels + c;
            if (bottom_data[bottom_index] > maxval) {
              maxval = bottom_data[bottom_index];
              maxidx = bottom_index;
            }
          }
        }
        output(b) = maxval;  //b代表 max pool 出来的有几个数
        argmax(b) = maxidx; //虽然操作的是展平的值，但实际上修改的还是没展平之前的
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pooled_height * pooled_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

```

* 关于Shard，请见[https://github.com/KeithYin/Faster-RCNN_TFpy2/blob/master/lib/roi_pooling_layer/work_sharder.h](https://github.com/KeithYin/Faster-RCNN_TFpy2/blob/master/lib/roi_pooling_layer/work_sharder.h)



 