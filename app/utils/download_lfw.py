from sklearn.datasets import fetch_lfw_people
import os

output_dir = "data/lfw"
os.makedirs(output_dir, exist_ok=True)

lfw = fetch_lfw_people(data_home=output_dir, download_if_missing=True)

print("Download concluído.")
print("Quantidade de imagens:", len(lfw.images))
print("Formato das imagens:", lfw.images.shape)
print("Quantidade de pessoas:", len(lfw.target_names))