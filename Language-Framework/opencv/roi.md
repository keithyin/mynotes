# 如何设置ROI

## 通过Rect设置

`Rect:` 表示一个矩形的一个类（基类）,构造函数是这样：
```
//(x,y)表示起始点.表示的矩形范围是(x:x+width, y:y+height)
Rect(int x, int y, int width, int height);
```

```c++
Mat roi = img(Rect(x, y, width, height));
```

## 通过Range设置

`Range:`表示范围的一个类（基类），构造函数是这样：
```
Range(int start, int end);
```

```c++
Mat roi = img(Range(begin_row, end_row), Range(begin_col, end_row))

```

**`roi`中保存的是ROI区域的指针，改变`roi`等价于改变原图。**

**opencv中一会(x,y)，一会(row, col)，好恼火！！！**
