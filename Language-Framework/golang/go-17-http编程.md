# 使用http上传文件

https://stackoverflow.com/questions/3508338/what-is-the-boundary-in-multipart-form-data

```go
func UploadFile(filename string, bow *browser.Browser) {
	bodyBuf := bytes.NewBufferString("")
	bodyWriter := multipart.NewWriter(bodyBuf)
	// 这个也贼重要, 不过也不知道为啥
	filewriter, _ := bodyWriter.CreateFormFile("multi_add_file", filename)
	fileHandler, _ := os.Open(filename)
	io.Copy(filewriter, fileHandler)
  
  // 这个非常重要!!!!! 不然上传不上去的, 不知道为啥....
	// 但是这个玩意会将 boundry 补全. boundry 是用来区分 k-value 对的.
  bodyWriter.Close() 
	contentType := bodyWriter.FormDataContentType()
	fmt.Println("content_type: ", contentType)
	fmt.Println(bodyBuf.String())
	err := bow.Post(someURL, contentType, bodyBuf)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(bow.StatusCode())
	fmt.Println(bow.Body())
}
```

