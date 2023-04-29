import faiss
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import time
import random
from datetime import datetime
import wget
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import requests
from tqdm import tqdm
import fire
from img2dataset import download
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import save_image_grid
import gc

script_path = os.path.dirname(os.path.realpath(__file__))

"""
This class is used to deduplicate images in the Laion dataset.
Performs Semantic Deduplication according to the paper: SemDeDup: Data-efficient learning at web-scale through semantic deduplication, https://arxiv.org/abs/2303.09540
"""
class SemDeDup:
    def __init__(
        self, 
        metadata_dir, 
        img_emb_dir, 
        num_clusters=100, 
        target_cluster_size = None, 
        epsilon=0.325, 
        cluster_process_thread_count=12,
        log_clusters=True,
        log_clusters_n=5,
    ):
      metadata = self.read_metadata(metadata_dir)
      img_emb = self.read_img_emb(img_emb_dir)
      
      assert metadata.shape[0] == img_emb.shape[0], "metadata and img_emb have different number of rows"
      print("Starting metadata size:", metadata.shape[0], '\n')
      
      self.metadata = metadata
      self.img_emb = img_emb
      
      if target_cluster_size:
        self.num_clusters = int(self.metadata.shape[0] / target_cluster_size)
        print("Setting num_clusters to", self.num_clusters, "based on target_cluster_size")
      else:
        self.num_clusters = num_clusters

      self.epsilon = epsilon
      self.cluster_process_thread_count = cluster_process_thread_count
      
      self.log_clusters = log_clusters
      self.log_clusters_n = log_clusters_n
      self.run_id = datetime.timestamp(datetime.now())

    def read_metadata(self, metadata_dir):
      """Reads dataset metadata parquest files from a directory.
      """
      assert metadata_dir is not None, "metadata_dir is None"

      def read_single_metadata_file(p):
          table = pq.read_table(os.path.join(metadata_dir, p))
          return table

      metadata_paths = sorted([p for p in os.listdir(metadata_dir) if p.endswith(".parquet")])
      
      print(f"Reading {len(metadata_paths)} metadata files...")

      with ThreadPoolExecutor() as executor:
          tables = list(executor.map(read_single_metadata_file, metadata_paths))

      metadata_table = pa.concat_tables(tables)
      metadata_df = metadata_table.to_pandas()
      return metadata_df
    

    def read_img_emb(self, img_emb_dir):
      """ Reads image embeddings .npy files from a directory.
      """
      assert img_emb_dir is not None, "img_emb_dir is None"

      def read_single_img_emb_file(p):
          np_array = np.load(os.path.join(img_emb_dir, p))
          return np_array

      img_emb_paths = sorted([p for p in os.listdir(img_emb_dir) if p.endswith(".npy")])

      print(f"Reading {len(img_emb_paths)} embedding files...")

      with ThreadPoolExecutor() as executor:
          np_arrays = list(executor.map(read_single_img_emb_file, img_emb_paths))

      img_emb = np.concatenate(np_arrays)
      return img_emb

    def create_index(self, embeddings):
      """ Creates a faiss index of embeddings.
      
      Args:
        embeddings: numpy array of embeddings
        
      Returns:
        index: faiss index of embeddings
      """
      index = faiss.IndexFlatIP(embeddings.shape[1])
      index.add(embeddings)
      return index

    def unbiased_sampling(self, embeddings, num_samples):
      num_embeddings = embeddings.shape[0]
      random_state = np.random.RandomState(123)
      indices = random_state.choice(num_embeddings, size=num_samples, replace=False)
      sampled_embeddings = embeddings[indices]

      return sampled_embeddings
  
    def train_kmeans(self, embeddings):
      
      max_training_points = 500000 # Protect against memory running out.
      kmeans = faiss.Kmeans(embeddings.shape[1], self.num_clusters, niter=20, verbose=True)
            
      if embeddings.shape[0] > max_training_points:
        sampled_embeddings = self.unbiased_sampling(embeddings, 500000)
        kmeans.train(sampled_embeddings)
      else:
        kmeans.train(embeddings)
      
      return kmeans
      
    def filter_cluster(self, cluster, cluster_index, epsilon):
      """ Deduplicate a cluster of embeddings.
      Compute all pairwise cosine similarities within a cluster.
      Groups embeddings within cluster into semantic duplicates when similarity > 1 - epsilon. 
      Keeps one random embedding from each group.
      
      Args:
        cluster: embeddings of images in the cluster
        cluster_index: faiss index of the cluster
        epsilon: threshold for cosine similarity
      Returns:
        keep_indices: indices of embeddings to keep from the cluster
      """
      
      cluster_distances, cluster_indices = cluster_index.search(cluster, cluster.shape[0])
      
      past_threshold = cluster_distances >= (1 - epsilon)
      assigned = set()
      groups = []
      
      for i, row in enumerate(cluster_indices):
        duplicate_indices = row[past_threshold[i]]
        index = duplicate_indices[0] 
        if index not in assigned:
          duplicate_indices = set(duplicate_indices)
          assigned = assigned.union(duplicate_indices)
          groups.append(duplicate_indices)
          
      # Randomly select one image from each group to keep
      # Alternate methods are choosing image closest/furthest to centroid, paper states negligible difference.
      keep_indices = np.array([random.choice(list(group)) for group in groups])

      return keep_indices

    def deduplicate(self):
      print("Starting deduplication...")
      kmeans_cluster = self.train_kmeans(self.img_emb)

      D, cluster_labels = kmeans_cluster.assign(self.img_emb)

      embeddings_to_keep = np.zeros(self.img_emb.shape[0], dtype=bool)

      visualize_clusters = random.sample(range(self.num_clusters), self.log_clusters_n)

      print(f"\n{self.num_clusters} clusters to process")
      
      def process_cluster(i):
        img_emb_cluster_indices = np.where(cluster_labels == i)[0]
        cluster = self.img_emb[img_emb_cluster_indices]

        centroid = kmeans_cluster.centroids[i]
        
        cluster_index = self.create_index(cluster)
        
        if self.log_clusters and i in visualize_clusters:
          self.visualize_cluster(cluster_index, img_emb_cluster_indices, centroid, f"cluster_{i}_before_dedup")

        keep_indices = self.filter_cluster(cluster, cluster_index, epsilon=self.epsilon)

        cluster_size = cluster.shape[0]
        remaining_proportion = keep_indices.shape[0] / cluster.shape[0]
        
        img_emb_indices = img_emb_cluster_indices[keep_indices]

        if self.log_clusters and i in visualize_clusters:
          filtered_cluster = cluster[keep_indices]
          filtered_cluster_index = self.create_index(filtered_cluster)
          self.visualize_cluster(filtered_cluster_index, img_emb_cluster_indices, centroid, f"cluster_{i}_after_dedup")
        
        
        return (i, img_emb_indices, remaining_proportion, cluster_size)

      # Process clusters in parallel.
      with ThreadPoolExecutor(max_workers=self.cluster_process_thread_count) as executor:
        futures = [executor.submit(process_cluster, i) for i in range(self.num_clusters)]

        with tqdm(total=len(futures)) as pbar:
          for future in as_completed(futures):
            try:
              i, img_emb_indices, remaining_proportion, cluster_size = future.result()
              embeddings_to_keep[img_emb_indices] = True
              del img_emb_indices
              gc.collect()
              tqdm.write(f"Processed cluster {i}: Keeping {remaining_proportion * 100:.2f}% of original size {cluster_size}")
              pbar.update(1)
            except Exception as e:
              print(f"Error processing cluster {i}: {e}")

      return embeddings_to_keep

    def visualize_cluster(self, cluster_index, img_emb_cluster_indices, centroid, name, image_grid=(4, 4)):
      """Visualizes a cluster from a k-means clustering using an image grid.
      
      Args:
        cluster_index: An faiss index of the k-means cluster centroids.
        img_emb_cluster_indices: The indices of the cluster in the original image embedding array.
        centroid (numpy.ndarray): The centroid of the cluster to be visualized.
        name (str): The output filename of the image grid.
        image_grid (tuple, optional): The size of the image grid to be used for visualizing the cluster. Defaults to (4, 4).
      """
      
      _closest_centroid_D, closest_centroid_I = cluster_index.search(np.expand_dims(centroid, axis=0), image_grid[0] * image_grid[1])
      img_indices = img_emb_cluster_indices[closest_centroid_I[0]]
      img_urls = self.metadata.iloc[img_indices]["url"].values
              
      images = []
      for url in img_urls:
        try:
          images.append(Image.open(BytesIO(requests.get(url).content)).convert("RGB"))
        except Exception as e:
          pass
  
      save_image_grid(images, image_grid, f"{script_path}/samples/{self.run_id}/{name}.jpg")


    def run(self):
      print("Running SemDeDup, ID:", self.run_id)
      embeddings_to_keep = self.deduplicate()
      
      proportion_remaining = np.sum(embeddings_to_keep) / self.metadata.shape[0]

      print("\n\nDeduplication Complete.\n",)
      print(f"Proportion of dataset remaining:: {proportion_remaining * 100:.2f}%, Original size: {self.metadata.shape[0]}, Remaining: {np.sum(embeddings_to_keep)}")
  
      deduplicated_data = self.metadata.iloc[embeddings_to_keep]
      deduplicated_table = pa.Table.from_pandas(deduplicated_data)
      
      return deduplicated_table


def deduplicate(
  metadata_dir: str = "./dataset/metadata",
  img_emb_dir: str = "./dataset/img_emb",
  output_dir: str = './output',
  target_cluster_size: int = 7500,
  num_clusters: int = 100,
  epsilon: float = 0.325,
  cluster_process_thread_count: int = 8,
  log_clusters: bool = True,
  log_clusters_n: int = 5,
  download_images: bool = True,
  download_image_size: int = 512,
  download_process_count: int = 8,
):
  start = time.time()

  semdedup = SemDeDup(
    metadata_dir=metadata_dir,
    img_emb_dir=img_emb_dir,
    target_cluster_size=target_cluster_size, 
    num_clusters=num_clusters, 
    epsilon=epsilon,
    cluster_process_thread_count=cluster_process_thread_count,
    log_clusters=log_clusters,
    log_clusters_n=log_clusters_n,
  )
  
  deduplicated_table = semdedup.run()

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  # Write output parquet metadata
  output_parquet_path = os.path.join(output_dir, 'deduplicated.parquet')
  pq.write_table(deduplicated_table, output_parquet_path)
  
  cluster_finish_time = time.time()
  print("Processed clusters in:", cluster_finish_time - start, "seconds")
  
  # Begin download of deduplicated dataset
  if download_images:
    download(
      processes_count=download_process_count,
      url_list=output_parquet_path,
      image_size=download_image_size,
      output_folder=output_dir,
      output_format="files",
      input_format="parquet",
      url_col="url",
      caption_col="caption",
      number_sample_per_shard=10000,
      distributor="multiprocessing",
    )
   
    end = time.time()
    print("Downloaded in:", cluster_finish_time - start, "seconds")
    print("Finished in:", end - start, "seconds")
      

def main():
  fire.Fire(deduplicate)

if __name__ == "__main__":
  main()

