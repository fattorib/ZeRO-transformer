from google.cloud import storage
from google.cloud.exceptions import NotFound

client = storage.Client()

# bucket = client.get_bucket(f"gs://bfattoriwebtext2")
# print(bucket)
# blobs = bucket.list_blobs(prefix=f"checkpoints")
# for blob in blobs:
#     blob.delete()

for blob in client.list_blobs(bucket = f"gs://bfattoriwebtext2", prefix=f"checkpoints/"):
    print(blob)