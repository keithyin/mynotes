# python GRPC

## 安装

```shell
python -m pip install grpcio
python -m pip install grpcio-tools
```



## 下载 example

```shell
git clone https://github.com/grpc/grpc
cd grpc/examples/python/helloworld
```



## 执行 grpc 程序

```shell
# 开启服务
python greeter_server.py
# 发请求
python greeter_client.py
```



## 更新 grpc 服务

```shell
cd grpc/examples/protos
# 修改 helloworld.proto
```

```protobuf
// The greeting service definition.
service Greeter {
  // Sends a greeting
  rpc SayHello (HelloRequest) returns (HelloReply) {}
  // Sends another greeting
  rpc SayHelloAgain (HelloRequest) returns (HelloReply) {}
}

// The request message containing the user's name.
message HelloRequest {
  string name = 1;
}

// The response message containing the greetings
message HelloReply {
  string message = 1;
}
```



## 重新生成grpc代码

```shell
cd grpc/examples/python/helloworld
python -m grpc_tools.protoc -I../../protos --python_out=. --grpc_python_out=. ../../protos/helloworld.proto

```

* 可以看出生成的两个文件
  * `helloworld_pb2.py`: 负责请求和响应的数据的序列化与反序列化
  * `helloworld_pb2_grpc.py`: 负责网络通信相关事宜

## 更新我们的应用程序

```python
# SERVER
class Greeter(helloworld_pb2_grpc.GreeterServicer):
  	# 继承 Servicer, 然后实现rpc接口
  	def SayHello(self, request, context):
    		return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

  	def SayHelloAgain(self, request, context):
    		return helloworld_pb2.HelloReply(message='Hello again, %s!' % request.name)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

```python
#CLIENT
def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = helloworld_pb2_grpc.GreeterStub(channel)
    # stub 直接调用 rpc 接口, 传入数据
    response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
    print("Greeter client received: " + response.message)
    response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='you'))
    print("Greeter client received: " + response.message)
```

