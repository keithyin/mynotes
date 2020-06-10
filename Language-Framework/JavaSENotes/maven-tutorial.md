# maven

* build tool
* Project management tool



* 下载与配置
  * 下载网址 https://maven.apache.org/download.cgi
  * 解压, 然后配置 
    * `export M2_HOME=~/Program/apache-maven-3.6.3`
    * `export PATH=$M2_HOME/bin:$PATH`
  * 然后 `mvn --version` 可以查看 `maven` 的版本信息.



## maven as build tool

* multiple jars: 新的项目可能要包含好多jar, 但是有时候我们不知道所有的jar是啥
* dependencies and versions: 依赖的jar同时依赖其它jar
* project structure
* building, publishing and deploying



### project structure

```shell
mkdir myapp
cd myapp
mvn archetype:generate
# 选择 archetype
# groupid: package 的 name
# artifactid: 项目名字?
```

* 编译

```shell
mvn compile
mvn package # package 成一个 jar 文件
```



