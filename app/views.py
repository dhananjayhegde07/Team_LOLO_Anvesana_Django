from django.shortcuts import render
from django.http import JsonResponse
from object_detector.settings import BASE_DIR
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from PIL import Image
model = tf.keras.models.load_model(f'{BASE_DIR}/models/component_classifier.h5')


def dsbjs(req):
    print(req.method)
    return JsonResponse({
        "status":'done'
    })

@csrf_exempt
def testing(request):
    if request.method == 'POST' and 'file' in request.FILES:
        # Retrieve the file from the request
        uploaded_file = request.FILES['file']
        # Load the image
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((150, 150))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)
        print(f"The predicted class is: {predicted_class}")
        class_indices = open(f'{BASE_DIR}/models/class_indices.txt').read()
        class_indices=class_indices.split(' ')
        
        predicted_label = class_indices[predicted_class[0]]
        links=open(f'{BASE_DIR}/models/class_links.txt').read().split(' ')
        print(f"The predicted class is: {predicted_label}")
        return JsonResponse({'status':'done',"class":predicted_label,'link':links[predicted_class[0]]})

    else:
        return render(request=request,template_name='index.html')