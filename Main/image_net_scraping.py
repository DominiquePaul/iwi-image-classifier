import numpy as np
from preprocessing import create_imagenet_dataset, create_imagenet_dataset_random

SIZE = 2000

synset_ids = ["n03051540", "n03051540"]
object_names = ["fashion","automotive"]

for synset_id, object in zip(synset_ids, object_names):
    print("Starting to process the {} dataset".format(object))

    imgs_object = create_imagenet_dataset(synset_id=synset_id, size=SIZE, use_child_synsets=True)
    imgs_random = create_imagenet_dataset_random(size=SIZE, max_synset_imgs=20, forbidden_synset=synset_id, exclude_synset_children=True)

    imgnet_imgs = np.concatenate((imgs_object, imgs_random))
    imgnet_labels = np.array([1]*len(imgs_object) + [0]*len(imgs_random))

    random_order = np.random.permutation(len(imgnet_imgs))
    imgnet_imgs = imgnet_imgs[random_order]
    imgnet_labels = imgnet_labels[random_order]

    np.save("imgnet_{}_x".format(object), imgnet_imgs)
    np.save("imgnet_{}_y".format(object), imgnet_labels)
    print("Finished processing the {} dataset".format(object))
