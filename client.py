import grpc
from protos import Server_pb2
from protos import Server_pb2_grpc

def run():
    print("Send Hello")
    # 替换为服务器的实际IP地址
    with grpc.insecure_channel('192.168.50.115:50051') as channel:
        print("Send Hello")
        stub = Server_pb2_grpc.ServerStub(channel)
        response = stub.ACallP_sample_align(Server_pb2.ServerRequest(message='sample_align'))
        print("Server client received: " + response.message)

if __name__ == '__main__':
    run()