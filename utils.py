from dataloader import ImageDataset, convert_dimensions
from torch import cat
from numpy import asarray, ones, zeros

import random

def generate_real_samples(dataset, n_samples):
    output_tensor = None
    for idx, x in enumerate(dataset):
        if idx >= n_samples:
            break
        if output_tensor == None:
            output_tensor = x
        else:
            output_tensor = cat((output_tensor, x), 0)

    return convert_dimensions(output_tensor.numpy()), ones((n_samples))
        
    pass

def generate_fake_samples(g_model, dataset):

    X = g_model(dataset)
    
    y = zeros(len(X))
    
    return X, y

def update_image_pool(pool, images, max_size = 50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            selected.append(image)
            pool.append(image)
            
        elif random() < 0.5:
            selected.append(image)
        else:
            idx = random.randint(0, len(pool))
            selected.append(pool[idx])
            pool[idx] = image
         
    return asarray(selected)
            