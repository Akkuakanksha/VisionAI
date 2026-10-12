import ssl
import tarfile
import urllib.request
from pathlib import Path

url = 'https://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz'
root = Path(__file__).resolve().parent
archive = root / 'ssd_mobilenet_v1_coco_11_06_2017.tar.gz'

ctx = ssl._create_unverified_context()
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as response, open(archive, 'wb') as out:
    out.write(response.read())

with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(root)

print(f'Downloaded and extracted to {root}')
