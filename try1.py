from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    data = load_files(path)
    plant_files = np.array(data['filenames'])
    plant_targets = np_utils.to_categorical(np.array(data['target']), 120)
    return plant_files, plant_targets

train_files, train_targets = load_dataset('C:/Users/Mahtab Noor Shaan/PycharmProjects/plant_seed_classification/new_train')
valid_files, valid_targets = load_dataset('C:/Users/Mahtab Noor Shaan/PycharmProjects/plant_seed_classification/new_validation')
#test_files, test_targets = load_dataset('C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition/test')

plant_names = [item[20:-1] for item in sorted(glob("C:/Users/Mahtab Noor Shaan/PycharmProjects/plant_seed_classification/new_train/*/"))]

# Let's check the dataset
print('There are %d total plant categories.' % len(plant_names))
print('There are %s total plant images.\n' % len(np.hstack([train_files, valid_files])))
print('There are %d training plant images.' % len(train_files))
print('There are %d validation plant images.' % len(valid_files))

print(train_targets[1])
print(train_files[1])
print(valid_targets[5])
print(valid_files[5])

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

train_vgg19 = VGG19(weights='imagenet', include_top=False).predict(preprocess_input_vgg19(train_tensors), batch_size=8)
valid_vgg19 = VGG19(weights='imagenet', include_top=False).predict(preprocess_input_vgg19(valid_tensors), batch_size=8)
train_resnet50 = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input_resnet50(train_tensors), batch_size=8)
valid_resnet50 = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input_resnet50(valid_tensors), batch_size=8)

np.save(open('train_vgg19.npy', 'wb'), train_vgg19)
np.save(open('valid_vgg19.npy', 'wb'), valid_vgg19)
np.save(open('train_resnet50.npy', 'wb'), train_resnet50)
np.save(open('valid_resnet50.npy', 'wb'), valid_resnet50)




