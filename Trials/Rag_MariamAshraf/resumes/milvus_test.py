from pymilvus import connections

try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus successfully!")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
