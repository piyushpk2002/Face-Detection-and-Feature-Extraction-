from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy import asarray, expand_dims
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def extract_face(filename, required_size=(160, 160)):
    # Load image from file
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    # Create the detector, 
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(pixels)
    print("results: ", results)
    
    if len(results) == 0:
        raise Exception("No faces found in the image.")

    # creating a plot
    fig, ax = plt.subplots()
    ax.imshow(image)
    for result in results:
        x, y, width, height = result['box']
       

        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        
    plt.axis('off')
    plt.title('Detected Faces')
    plt.show()

    # Extract bounding box from the first face in the image
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # Extract the face
    face = pixels[y1:y2, x1:x2]
    
    # Resize pixels to the model size

    # this line generates image from an array of pixels
    image = Image.fromarray(face)
    #resizes image to its original size
    image = image.resize(required_size)

    face_array = asarray(image)

    return face_array


image_path = 'sample.jpg'  # importing image

# Detect and extract face
face = extract_face(image_path)
print("Face extracted:", face.shape)

#Loading  FaceNet model using keras-facenet
embedder = FaceNet()
print("FaceNet model loaded.")


# The embedder expects a batch of faces, so pass [face]
embedding = embedder.embeddings([face])[0]
print("Embedding shape:", embedding.shape)
print("Embedding vector (first 10 values):", embedding[:10])


plt.figure(figsize=(12, 4))
plt.bar(range(len(embedding)), embedding)
plt.title('FaceNet Embedding (128-D Features)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()
