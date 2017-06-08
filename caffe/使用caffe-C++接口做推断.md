# 使用caffe c++接口做推断

## 初始化网络

```c++
/************************初始化网络**************************/  
#include "caffe/caffe.hpp"  
#include <string>  
#include <vector>
using namespace caffe;  
using namespace std;

string proto = "H:\\Models\\Caffe\\deploy.prototxt"; /* 加载CaffeNet的配置 */  
Phase phase = TEST; /* or TRAIN */  
Caffe::set_mode(Caffe::CPU);  
// Caffe::set_mode(Caffe::GPU);  
// Caffe::SetDevice(0);  
  
//! Note: 后文所有提到的net，都是这个net  
boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, phase));  
/************************加载已训练好的模型**************************/
string model = "H:\\Models\\Caffe\\bvlc_reference_caffenet.caffemodel";      
net->CopyTrainedLayersFrom(model);


/*******************读取模型中的每层的结构配置参数*************************/   
NetParameter param; //保存模型参数
ReadNetParamsFromBinaryFileOrDie(model, &param); 
int num_layers = param.layer_size();  
for (int i = 0; i < num_layers; ++i)  
{  
    // 结构配置参数:name，type，kernel size，pad，stride等  
    LOG(ERROR) << "Layer " << i << ":" << param.layer(i).name() << "\t" << param.layer(i).type();  
    if (param.layer(i).type() == "Convolution")  
    {  
        ConvolutionParameter conv_param = param.layer(i).convolution_param();  
        LOG(ERROR) << "\t\tkernel size: " << conv_param.kernel_size()  
            << ", pad: " << conv_param.pad()  
            << ", stride: " << conv_param.stride();  
    }  
}  
```



## 读取图像均值

```c++
string mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";  
Blob<float> image_mean;  
BlobProto blob_proto; //缓存从mean_file中读取的数据
const float *mean_ptr;  
unsigned int num_pixel;  
  
bool succeed = ReadProtoFromBinaryFile(mean_file, &blob_proto);  
if (succeed)  
{  
    image_mean.FromProto(blob_proto);  
    num_pixel = image_mean.count(); /* NCHW=1x3x256x256=196608 */  
    mean_ptr = (const float *) image_mean.cpu_data();  
}  
```

* A Blob is a wrapper over the actual data being processed and passed along by Caffe.
  * Mathematically, a blob is an N-dimensional array stored in a C-contiguous fashion.
* 这里需要注意的几个函数是：
  * `ReadProtoFromBinaryFile` 从二进制文件中读`proto` 



## 前向过程

```c++
//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据  
void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)  
{  
    Blob<float>* input_blobs = net->input_blobs()[0];
    switch (Caffe::mode())  
    {  
    case Caffe::CPU:  
        memcpy(input_blobs->mutable_cpu_data(), data_ptr,  
            sizeof(float) * input_blobs->count());  
        break;  
    case Caffe::GPU:  
        cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,  
            sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);  
        break;  
    default:  
        LOG(FATAL) << "Unknown Caffe mode.";  
    }   
    net->ForwardPrefilled();  
}  
```



## 根据feature层的名字获取其在网络中的index

```c++
/*********************根据Feature层的名字获取其在网络中的Index******************/  
  
//! Note: Net的Blob是指，每个层的输出数据，即Feature Maps  
// char *query_blob_name = "conv1";  
unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)  
{  
    std::string str_query(query_blob_name);      
    vector< string > const & blob_names = net->blob_names();  
    for( unsigned int i = 0; i != blob_names.size(); ++i )   
    {   
        if( str_query == blob_names[i] )   
        {   
            return i;  
        }   
    }  
    LOG(FATAL) << "Unknown blob name: " << str_query;  
}
```



## 读取feature层的数据

```c++
/*********************读取网络指定Feature层数据*****************/  
  
//! Note: 根据CaffeNet的deploy.prototxt文件，该Net共有15个Blob，从data一直到prob      
char *query_blob_name = "conv1"; /* data, conv1, pool1, norm1, fc6, prob, etc */  
unsigned int blob_id = get_blob_index(net, query_blob_name);  
  
boost::shared_ptr<Blob<float> > blob = net->blobs()[blob_id];  
unsigned int num_data = blob->count(); /* NCHW=10x96x55x55 */  
const float *blob_ptr = (const float *) blob->cpu_data();  
```

* `net`的 `blob` 是 `feature maps`
* `Layer` 的 `blob` 是权重和偏置

