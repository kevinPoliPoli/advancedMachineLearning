
from typing import SupportsInt
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import sys
from tqdm import tqdm
import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
      
def augment_pipeline(sample):
  transformation1 = transforms.Compose(transforms.RandomHorizontalFlip(p=1))
  transformation2 = transforms.Compose(transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)))
  transformation3 = transforms.Compose(transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),transforms.RandomHorizontalFlip(p=1))
  sample1 = transformation1(sample)
  sample2 = transformation1(sample)
  sample3 = transformation1(sample)
  return sample1, sample2, sample3


dictionary = {
  'lamp' : 0,
  'umbrella' : 1,
  'Motorbikes' : 2,
  'butterfly' : 3,
  'Faces_easy' : 4,
  'ferry' : 5,
  'camera' : 6,
  'dollar_bill' : 7,
  'starfish' : 8,
  'stop_sign' : 9,
  'binocular' : 10,
  'Leopards' : 11,
  'snoopy' : 12,
  'saxophone' : 13,
  'pigeon' : 14,
  'headphone' : 15,
  'buddha' : 16,
  'Faces' : 17,
  'crocodile_head' : 18,
  'bass' : 19,
  'crocodile' : 20,
  'tick' : 21,
  'cougar_face' : 22,
  'ant' : 23,
  'cannon' : 24,
  'lobster' : 25,
  'watch' : 26,
  'dalmatian' : 27,
  'euphonium' : 28,
  'flamingo_head' : 29,
  'octopus' : 30,
  'panda' : 31,
  'cellphone' : 32,
  'ceiling_fan' : 33,
  'pagoda' : 34,
  'schooner' : 35,
  'gramophone' : 36,
  'mandolin' : 37,
  'electric_guitar' : 38,
  'ibis' : 39,
  'dragonfly' : 40,
  'soccer_ball' : 41,
  'anchor' : 42,
  'platypus' : 43,
  'yin_yang' : 44,
  'rhino' : 45,
  'menorah' : 46,
  'sea_horse' : 47,
  'wild_cat' : 48,
  'barrel' : 49,
  'beaver' : 50,
  'lotus' : 51,
  'stegosaurus' : 52,
  'llama' : 53,
  'nautilus' : 54,
  'strawberry' : 55,
  'bonsai' : 56,
  'emu' : 57,
  'accordion' : 58,
  'rooster' : 59,
  'garfield' : 60,
  'hedgehog' : 61,
  'cup' : 62,
  'ketch' : 63,
  'pizza' : 64,
  'elephant' : 65,
  'helicopter' : 66,
  'crayfish' : 67,
  'scorpion' : 68,
  'windsor_chair' : 69,
  'stapler' : 70,
  'joshua_tree' : 71,
  'revolver' : 72,
  'chair' : 73,
  'okapi' : 74,
  'car_side' : 75,
  'hawksbill' : 76,
  'inline_skate' : 77,
  'scissors' : 78,
  'sunflower' : 79,
  'airplanes' : 80,
  'laptop' : 81,
  'pyramid' : 82,
  'wheelchair' : 83,
  'mayfly' : 84,
  'minaret' : 85,
  'brain' : 86,
  'dolphin' : 87,
  'wrench' : 88,
  'brontosaurus' : 89,
  'gerenuk' : 90,
  'grand_piano' : 91,
  'cougar_body' : 92,
  'water_lilly' : 93,
  'flamingo' : 94,
  'chandelier' : 95,
  'kangaroo' : 96,
  'crab' : 97,
  'ewer' : 98,
  'trilobite' : 99,
  'metronome' : 100
}


class Caltech(VisionDataset):

    dataset = []

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = str(split) # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.dataset = []
        
        import shutil
        temp = open("temp.txt", "w")
        shutil.copyfile("/content/Caltech101/"+split+".txt",temp)
        temp.close()

        print("reading "+split+".txt ...")
        num_lines = sum(1 for line in open("/content/Caltech101/"+split+".txt",'r'))
        with open(temp,'r') as f:
          for line in tqdm(f, total=num_lines):
            line = line.strip()
            className = line.split('/')[0].strip()
            imageName = line.split('/')[1].strip()
            for subFolder in os.listdir(root):
              if (subFolder != "BACKGROUND_Google" and subFolder==className):
                for image in os.listdir("/content/Caltech101/101_ObjectCategories/"+subFolder):
                  if (image.__eq__(imageName)):
                    converted = pil_loader("/content/Caltech101/101_ObjectCategories/"+subFolder+"/"+image)
                    transformed = transform(converted)
                    self.dataset.append((transformed, dictionary[className]))
                    if augment:
                      sample1, sample2, sample3 = augment_pipeline(transformed)
                      self.dataset.append((sample1, dictionary[className]))
                      self.dataset.append((sample2, dictionary[className]))
                      self.dataset.append((sample3, dictionary[className]))
                      temp.append(subFolder + "/" + image)
                      temp.append(subFolder + "/" + image)
                      temp.append(subFolder + "/" + image)

          
        '''
        - Here you should implement the logic for reading the splits files and accessing elem
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
    def __getDataset__(self):
      return self.dataset

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image, label = self.dataset[index][0], self.dataset[index][1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.dataset)

        return length