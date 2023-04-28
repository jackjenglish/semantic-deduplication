import wget
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

MAX_SHARDS = 2304

def download_embedding_file(i, output_dir):
    shard_url = f'https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/img_emb/img_emb_{"{:04d}".format(i)}.npy'
    emb_filename = os.path.join(output_dir, f'img_emb/img_emb_{i:04d}.npy')
    
    if not os.path.exists(emb_filename):
      print('Downloading', emb_filename)
      wget.download(shard_url, emb_filename)

def download_embeddings(num_shards=1, output_dir='./dataset', max_workers=4):
    if not os.path.exists(os.path.join(output_dir, 'img_emb')):
        os.makedirs(os.path.join(output_dir, 'img_emb'))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda i: download_embedding_file(i, output_dir), range(num_shards))


def download_metadata_file(i, output_dir):
    metadata_url = f'https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_{"{:04d}".format(i)}.parquet'
    metadata_filename = os.path.join(output_dir, f'metadata/metadata_{i:04d}.parquet')
    
    if not os.path.exists(metadata_filename):
      print('Downloading', metadata_filename)
      wget.download(metadata_url, metadata_filename)

def download_metadata(num_shards=1, output_dir='./dataset', max_workers=4):
    if not os.path.exists(os.path.join(output_dir, 'metadata')):
        os.makedirs(os.path.join(output_dir, 'metadata'))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda i: download_metadata_file(i, output_dir), range(num_shards))

def install_img2dataset():
    print("Installing img2dataset...")
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Clone the repo.
    try:
        repo_url = "https://github.com/rom1504/img2dataset.git"
        clone_command = f"git clone {repo_url}"
        subprocess.run(clone_command, cwd=script_path, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning the repository: {e}")
        return
    # Run setup script.
    try:
        setup_command = "python setup.py install"
        subprocess.run(setup_command, cwd=os.path.join(script_path,"img2dataset"), check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing the package: {e}")
        return

    print("Successfully installed img2dataset.")

if __name__ == "__main__":
  download_embeddings(32)
  download_metadata(32)
  install_img2dataset()