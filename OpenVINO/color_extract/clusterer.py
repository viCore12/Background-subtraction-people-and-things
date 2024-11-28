import cv2
import numpy as np
import faiss
import matplotlib.pyplot as plt

class Clusterer:
    def __init__(self, n_cluster: int, d: int, n_iter: int):
        self.kmeans = faiss.Kmeans(d, n_cluster, niter=n_iter, verbose=False)
    def get_pixels(self, image, mask):
        output = image[mask > 0]
        return output
    def forward(self, image, mask):
        roi_pixels = self.get_pixels(image, mask)
        if len(roi_pixels) == 0:
            return None, None
        roi_pixels = roi_pixels.reshape((-1, 3))
        pixels = np.array(roi_pixels, dtype=np.float32)    
        self.kmeans.train(pixels)
        centroids = np.uint8(self.kmeans.centroids)
        distances, labels = self.kmeans.index.search(pixels, 1)
        return centroids, labels
        # res = centroids[labels.flatten()]
        # res = res.reshape(ori_shape)
        # return res
    def visualize(self, image, mask, centroids, labels, folder, index): 
        colors, counts = np.unique(labels, return_counts=True)
        percentages = counts / np.sum(counts)
        labels = [str(round(x * 100, 2)) for x in percentages]
        # Show original image
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Image")
        
        # Show percentages
        plt.subplot(1, 2, 2)
        colors=[tuple(x / 255.0) for x in centroids[colors]]
        plt.pie(percentages, labels=labels, colors=colors)
        plt.tight_layout()
        plt.savefig(f"{folder}/{index}.png")
        
