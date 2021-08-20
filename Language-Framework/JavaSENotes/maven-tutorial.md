# maven

* build tool
* Project management tool



如果不用maven，项目开发中存在的问题：[Why]

1. 一个项目就是一个工程
   1. 如果一个项目非常庞大，就不适合继续使用 `package` 来划分模块，最好是一个模块对应一个工程，利于分工协作。
   2. 利用 maven 可以将一个项目拆分成多个工程
2. 项目中需要的jar包必须手动 复制粘贴到 工程目录下
   1. 如果同样的jar包出现在不同的项目中，我们需要复制粘贴多份，比较浪费空间。
3. jar包需要别人准备好，或者官网下载
   1. 借助maven，可以已一种规范的方式下载jar包
4. 一个jar包依赖的其它jar包也需要自己手动加入到项目中
   1. maven会自动将依赖的jar包引入进来



Maven是什么 [What]

* 服务于 **java平台** 的自动化 **构建** 工具
  * Make -> Ant -> Maven -> Gradle
  * 构建工具：以 `java 源文件，框架配置文件，JSP，HTML，图片` 等资源为原材料，去生产一个可以运行的项目的过程
  * 编译：java源文件->编译->class字节码->交给jvm执行
  * 部署：如果是web工程，将编译的结果放到服务器上，就是部署
  * 搭建：
* 构建过程中的各个环节
  * 清理
  * 编译
  * 测试：自动调用 Junit 程序
  * 报告：测试程序执行的结果
  * 打包：动态web打war包，java工程打jar包
  * 安装：打包得到的文件复制到“仓库”的指定位置
  * 部署：动态web工程生成的war包复制到 servlet 容器指定的目录下，使其可以运行



* 下载与配置
  * 下载网址 https://maven.apache.org/download.cgi
  * 配置 `JAVA_HOME` 环境变量
  * 解压, 然后配置 
    * `export M2_HOME=~/Program/apache-maven-3.6.3`
    * `export PATH=$M2_HOME/bin:$PATH`
  * 然后 `mvn --version` 可以查看 `maven` 的版本信息.



> 本地仓库位置：`~\.m2\repository`
>
> 指定本地仓库位置的配置文件：`apache-maven-3.2.2\conf\settings.xml`
>
> 在根标签`settings`下添加如下内容：`<localRepository>本地仓库路径，也就是RepMaven.zip的解压目录</localRepository>`



maven核心概念

1. 约定的目录结构。

   1. ```shell
      Hello：项目名
      	src
      		main
      			java: 存放java源文件
      			resources: 存放框架或其它工具的配置文件
         	test
         		java
         		resources
      	pom.xml
      ```

   2. 

2. POM

3. 坐标：在主仓库中唯一定位一个工程

4. 依赖

   1. mvn解析依赖信息时会到本地 repo 中找

5. 仓库

   1. 本地仓库
   2. 远程仓库
      1. 私服：搭建在局域网环境中
      2. 中央仓库
      3. 中央仓库的镜像：为中央仓库分担流量
   3. 仓库中保存的内容
      1. Maven自身所需要的插件
      2. 第三方框架或工具的jar包
      3. 自己开发的Maven工程

6. 生命周期/插件/目标

7. 继承

8. 聚合



> Maven 核心程序仅仅定义了抽象的生命周期，但是具体的工作必须由特定的插件完成。而插件本身并不在maven核心程序中
>
> 当我们执行的maven命令需要某些插件时，maven核心程序会首先到本地仓库中查找
>
> 本地仓库地址 ~/.m2/repository
>
> 如果本地仓库没有，就需要联网去中央仓库下载
>
> * 如何修改 默认的本地仓库地址
>   * mvn 解压路径下的 `conf/settings.xml` 中有相应的配置





## pom.xml

> Project Object Model
>
> jar包的名字也是根据 坐标 来构成的。

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
  	<!-- 以下三个字段唯一标识一个maven工程！是在主仓库中唯一定一个maven工程-->
  	<groupId>com.yinpeng</groupId> <!-- 公司域名倒序+项目名-->
    <artifactId>SomeName</artifactId> <!-- 模块名称-->
    <version>0.0.1-SNAPSHOT</version>

    <name>ProjectName</name>
    <url>http://maven.apache.org</url>
  
  
  <dependencies>
  			<dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.10</version>
            <scope>test</scope>
        </dependency>
  </dependencies>
```



## 依赖范围

* Maven 项目主要分成两个部分，主程序 & 测试程序
  * compile：对应的是 主程序的依赖 & 测试程序的依赖，参与打包
    * 编译，测试，打包
  * test： 对应的是 测试程序的依赖，不参与打包
    * 测试
  * provided：web工程相关。开发时候使用，测试时使用。不参与打包
    * 编译，测试

```
compile
test
provided
```



## 生命周期

> 执行生命周期的某个阶段，会将前面的阶段都执行了
>
> 1. 生命周期的各个阶段仅仅是定义了要执行的任务是什么
> 2. 各个阶段和插件对应的目标是一致的
> 3. 相似的目标由特定的插件来完成

```
清理
编译
测试
报告
打包
安装
部署
```



## 依赖

* `test` 范围的依赖是不可以传递的。

* 依赖的传递性：依赖一个子模块，就不用care其依赖的模块了

* 依赖的排除：

  * 子模块引入的模块进行排除。（如果那个jar包不稳定的话，就可以去排除）

  * 排除了会不会影响原来的包？不会，因为之前的包已经是打包好的，排除的包的代码已经包含在里面了

  * ```xml
    <dependency>
    	<groupId>..</groupId>
      <artifactId>..</artifactId>
      <exclusions>
      	<exclusion>
          <groupId></groupId>
          <artifactId></artifactId> <!-- 这里不需要版本号-->
        </exclusion>
      </exclusions>
    <dependency>

* 依赖的原则：解决jar包冲突的问题。项目A依赖了B1，B2. B1，B2依赖了不同的 C版本。那么依赖的传递会传递给 A什么版本的C呢？

  * 就近优先原则，距离最短的优先。
  * 如果是相同的路径长度：先声明者优先，在 denpendency 中的声明位置
  * **没有按照 版本号来进行优先度评判**

* 统一管理依赖的版本

  * 在 `properties` 标签内使用自定义标签统一声明版本号

  * 在需要统一版本的位置，使用 `${label_name}` 引用声明的版本号

  * ```xml
    // pom.xml
    
    <properties>
    	<labelName>value</labelName>
    	<labelName2>value2</labelName2>
    </properties>
    
    <dependency>
    	<groupId>..</groupId>
      <artifactId>..</artifactId>
      <version>${labelName}</version>
    </dependency>
    ```



## 继承

* 统一管理版本依赖问题。

* 将junit依赖统一提取到父工程中，在子工程声明时不指定版本

* 父工程：创建一个maven工程作为父工程。注意，打包的方式为 pom

* 在子工程声明对父工程的引用

  * ```xml
    <!-- 子模块的 pom.xml -->
    
    <!--  -->
    
    <parent>
    	<groupId></groupId>
      <artifactId></artifactId>
      <version></version>
      
      <relativePath>../SomePath/pom.xml</relativePath> <!-- 父工程的pom相对路径-->
    </parent>
    ```

  * 

* 在父工程中统一 junit 依赖

  * ```xml
    <dependencyManagement>
    	<dependencies>
      	<dependency>
        	<groupId></groupId>
          <artifactId></artifactId>
          <version></version>
          <scope></scope>
        </dependency>
      
      </dependencies>
    </dependencyManagement>
    ```

  * 

* 在子工程删除对junit 的版本号



## 聚合

* 做了继承之后，一定要先安装 父工程，否则子工程无法安装

* 一键安装各个模块工程:  可以如此配置

  * ```xml
    <!-- 可以在父项目中配置-->
    <modules>
    	<module>../Path</module>
      <module>../Path2</module>
    </modules>
    ```

  * 

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



# 常用 mvn 命令

```shell
mvn clean #清理
mvn compile #编译
mvn test-compile #编译测试程序
mvn test #测试
mvn pakcage #打包



mvn install # 将当前模块安装到 本地 repo 中
```



