from django.shortcuts import render, HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import generics
from .serializers import SearchSerializers
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from PIL import Image
from skimage import io
from skimage.feature import SIFT
import numpy as np
from collections import Counter
import nmslib
from collections import Counter
import pickle

index = nmslib.init(method='hnsw', space='l2')
index.loadIndex(r"D:\RealCoderZ\ReverseImageSearch\PythonSIFT\index_9.bin")

with open(r"D:\RealCoderZ\ReverseImageSearch\desc_metadata9.pkl", 'rb') as f:
    img_name, original_indices, matrix = pickle.load(f)

detector_extractor = SIFT()

print('----------------Index with metadata loaded---------------')

def desc_cal():
    image = io.imread(r"static/input/file.jpg", as_gray = True) 
    detector_extractor.detect_and_extract(image)
    inp_img_desc = detector_extractor.descriptors
    return inp_img_desc

def nearestneighbor(inp_img_desc):
    all_indx = []
    for i in inp_img_desc:
        query = i    
        indices, _ = index.knnQuery(query, k=5)
        all_indx.append(indices)
    
    return all_indx

def original_image_index(all_indx):
    original_most = []
    for j in all_indx:
        og_most = []
        for i in j:
            og_most.append(original_indices[i])
        original_most.append(og_most)

    unique_list = []
    for i in original_most:
        u_list = []
        for j in i:
            if j not in u_list:
                u_list.append(j)
    unique_list.append(u_list)
    nf = np.concatenate(unique_list)
    return nf

def most_repeated_value(lst):
    count_dict = Counter(lst)
    most_repeated = sorted(count_dict.items(), key=lambda x: x[1], reverse = True)
    return most_repeated

@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'search_image': 'http://127.0.0.1:8000/engine/search/',
    }
    return Response(api_urls)

@api_view(['GET'])
def show_image_url(request, name):
    image_url = f'static/images9/{name}'
    with open(image_url, 'rb') as f:
        image = f.read()
    return HttpResponse(image, content_type='image/jpeg')

class SearchImage(generics.CreateAPIView):
    serializer_class = SearchSerializers

    def post(self, request):
        img_uploaded = request.FILES.get('image')
        default_storage.save('static/input/file.jpg', ContentFile(img_uploaded.read()))

        inp_img_desc = desc_cal()
        all_indx = nearestneighbor(inp_img_desc)
        nf = original_image_index(all_indx)
        mr = most_repeated_value(nf)
        print([img_name[i[0]] for i in mr[:5]])
        
        found = [f"http://127.0.0.1:8000/engine/extract_img/{img_name[i[0]]}" for i in mr[:3]]

        default_storage.delete('static/input/file.jpg')

        return Response(
            {
                'status': found
            }
        )