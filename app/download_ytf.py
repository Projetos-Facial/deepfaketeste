import os
import urllib.request
import tarfile
import tqdm
import ssl

# Ignorar o certificado de segurança não seguro da Universidade
ssl_context = ssl._create_unverified_context()

def download_file(url, output_path):
    print(f"📡 Iniciando download de: {url}")
    with urllib.request.urlopen(url, context=ssl_context) as response, open(output_path, 'wb') as out_file:
        file_size = int(response.info().get('Content-Length', -1))
        
        if file_size == -1:
            print("⏳ Baixando (tamanho desconhecido, aguarde...)")
            out_file.write(response.read())
        else:
            with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Baixando") as pbar:
                chunk_size = 1024 * 64
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk: break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
    print("✅ Download concluído!")

def setup_youtube_faces():
    data_dir = "data/ytf"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download do Arquivo de Splits (os pares que vamos testar)
    splits_url = "http://www.cs.tau.ac.il/~wolf/ytfaces/splits.txt"
    splits_path = os.path.join(data_dir, "splits.txt")
    if not os.path.exists(splits_path):
        download_file(splits_url, splits_path)
    else:
        print("✅ Arquivo splits.txt já existe.")

    # 2. Download dos Frames Alinhados (Imagens principais)
    # OBS: Pode ter 1.8GB, requer internet estável.
    tar_url = "https://www.cs.tau.ac.il/~wolf/ytfaces/aligned_images_DB.tar.gz"
    tar_path = os.path.join(data_dir, "aligned_images_DB.tar.gz")
    
    if not os.path.exists(tar_path):
        print("\n⏳ Atenção: O arquivo tem quase 2GB! Pode levar de 3 a 10 minutos.")
        download_file(tar_url, tar_path)
    else:
        print("✅ Arquivo aligned_images_DB.tar.gz já existe.")

    # 3. Extrair os frames
    extracted_dir = os.path.join(data_dir, "aligned_images_DB")
    if not os.path.exists(extracted_dir):
        print("\n📦 Extraindo 2GB de frames (Isso também demorará uns minutinhos)...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("✅ Extração completa!")
    else:
        print("✅ Imagens já estão extraídas e prontas!")
        
    print("\n🎉 YOUTUBE FACES DB PRONTO PARA USO!")
    print(f"Os arquivos estão em: {data_dir}/aligned_images_DB")

if __name__ == "__main__":
    setup_youtube_faces()
