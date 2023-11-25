from PIL import Image
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import skimage.io as io
from time import time
from mpi4py import MPI


# Function to read saved images
def read_images_from_directory_parallel(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = load_image(os.path.join(directory, filename))
            images.append(img)
    return images


# Load the comparison image
def load_image(image_path):
    return np.array(Image.open(image_path))


def compare_images(image1, image2):
    return ssim(image1, image2, channel_axis=2)


if __name__ == '__main__':
    start_time = time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_batches = 5
    # Each process reads its assigned batches
    dataset = []
    for batch_idx in range(rank, num_batches, size):
        batch_directory = f"cifar_images/cifar-10-batches-py/data_batch_{batch_idx + 1}"
        dataset.extend(read_images_from_directory_parallel(batch_directory))

    # Gather datasets from all processes to the root process
    all_datasets = comm.gather(dataset, root=0)

    # Distribute chunks of the combined dataset to all processes
    if rank == 0:
        # Combine datasets from all processes
        combined_dataset = [image for process_dataset in all_datasets for image in process_dataset]
        # Split the dataset into chunks for distribution
        chunks = np.array_split(combined_dataset, size)
    else:
        chunks = None

    # Scatter the chunks to all processes
    chunk = comm.scatter(chunks, root=0)

    # Load the test image in each process
    test_img = load_image('./test_image.png')

    # Each process finds its most similar image in its chunk
    local_max_similarity = 0
    local_similar_img = None
    for img in chunk:
        similarity = compare_images(img, test_img)
        if local_max_similarity < similarity:
            local_max_similarity = similarity
            local_similar_img = img

    # Gather the most similar images and their scores from all processes
    all_max_similarities = comm.gather(local_max_similarity, root=0)
    all_similar_imgs = comm.gather(local_similar_img, root=0)

    if rank == 0:
        # Find the globally most similar image
        global_max_similarity = max(all_max_similarities)
        global_max_index = all_max_similarities.index(global_max_similarity)
        global_similar_img = all_similar_imgs[global_max_index]
        # Save the globally most similar image
        if global_similar_img is not None:
            io.imsave("./similar_image.png", global_similar_img)

        print(f"Total time taken: {time() - start_time} seconds")