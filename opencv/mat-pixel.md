# 读取Mat中的像素
**Mat对象中保存了保存图像的矩阵的指针（只是指针哦）！**

**存放图像的矩阵是个二维的，shape为(rows, cols*channels)!**

**每行的像素排列是这样的，[c0,c1,c2, c0,c1,c2]**

## 使用指针访问像素
```c++
void read_pixel_from_img(Mat img){
  int rows = img.rows;
  int cols = img.cols;
  int pixel_per_row = cols*img.channels();
  //二维数组，两个for循环就足够
  for (int i=0; i<rows; i++){
    uchar* data = img.ptr<uchar>(i); // ptr<uchar>(int row); 返回指向行的指针
    for (int j=0; j<pixel_per_row; j++){
      uchar pixel = data[j];//读取到了单个像素
    }
  }
}
```

## 使用迭代器操作像素
先介绍一个类：

- VecNt, 其中N代表几channel,t表示是什么类型。表示有N个类型为t的元素 的 向量，Mat的子类。

迭代器就是从`begin`到`end`
```c++
void read_pixel_from_img(Mat img){
  Mat_<Vec3b>::iterator it = img.begin<Vec3b>(); // 一次搞出来三个值。
  Mat_<Vec3b>::iterator end = img.end<Vec3b>();
  for(; it!=end; ++it){
    //读取每个像素
    uchar b = (* it)[0];
    uchar g = (* it)[1];
    uchar r = (* it)[2];
  }
}
```

## 使用动态地址计算
```c++
int rows = img.rows;
int cols = img.cols;
    //int pixel_per_row = cols*img.channels();
    //二维数组，两个for循环就足够
for (int i=0; i<rows; i++){
  for (int j=0; j<cols; j++){
      //注意这里是小于cols，而不是pixel_per_row.
      Vec3b bgr = img.at<Vec3b>(i,j); // 这个就是一次读三个值的意思。
      uchar b = bgr[0];
      uchar g = bgr[1];
      uchar r = bgr[2];
  }
}
```





## 直接操作底层指针

```c++
Mat a = imread("/home/keith/Pictures/icon.png");
Mat processed = a.clone();
// .data 返回底层指针！！！
uchar *data = a.data;
uchar* processed_data = processed.data;
const long count = a.rows * a.cols * a.channels();
for(long i = 0; i < count; i++){
  *(processed_data+i) = (*(data+i))/5;
}
imshow("origin", a);
imshow("processed", processed);
```

