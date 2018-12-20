## 环境配置

```shell
# Properties -> Java Build Path -> Library -> Add lib -> User Lib -> new  
```

**添加哪些 jar包**

```shell
cd hadoop-3.1.1/share/hadoop/hdfs
# hadoop-hdfs-3.1.1.jar
# lib 下面的所有包

cd ../common
# hadoop-common-3.1.1.har
# lib 下面的所有包
```



```java
package keith.hdfs;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class HDFSUtils {
	public static void main(String[] args) throws IOException {
		
		// read the configuration file
		// copy core-site.xml hdfs-site.xml to current directory
		Configuration conf = new Configuration();
		
		// this is the client of HDFS
		FileSystem fs = FileSystem.get(conf);
		
		// Download
		
	}
	
	public void upload(FileSystem fs) throws IOException {
		Path des = new Path("hdfs://192.168.204.200:9000/file2.txt");
		FSDataOutputStream os = fs.create(des);
		FileInputStream in = new FileInputStream("file2.txt");
		IOUtils.copy(in, os);
		
		// fs.copyFromLocalFile(new Path("path1"), des);
	}
	
	public void download(FileSystem fs) throws IOException {
		Path src = new Path("hdfs://192.168.204.200:9000/file.txt");
		FSDataInputStream in = fs.open(src);
		FileOutputStream os = new FileOutputStream("/home/keith/download/file.txt");
		IOUtils.copy(in, os);
	}
	
	public void listFiles() {
		
	}
	public void mkdir(FileSystem fs) throws IllegalArgumentException, IOException {
		fs.mkdirs(new Path("/aaa/bbb/ccc"));
	}
	public void rm() {
		
	}

}
```



