import sys
import subprocess
import os

def swap_faces(source_img, target_media, output_media):
    print("\n" + "="*80)
    print(" 🎭 INICIANDO FACE SWAP (FACEFUSION) ".center(80, "═"))
    print("="*80)
    print(f" Origem (Rosto)  : {source_img}")
    print(f" Destino (Fundo/Vídeo): {target_media}")
    print(f" Saída           : {output_media}")
    print("\n⏳ Executando FaceFusion... (Isso pode levar alguns segundos)")
    
    # Chama o script do facefusion acionando o processamento
    cmd = [
        "python",
        "facefusion.py",
        "headless-run",
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "-s", os.path.abspath(source_img),
        "-t", os.path.abspath(target_media),
        "-o", os.path.abspath(output_media),
        "--execution-providers", "cuda"
    ]
    
    try:
        subprocess.run(cmd, cwd="facefusion", check=True)
        print("\n✅ Face Swap concluído com sucesso!")
        print(f"📁 Arquivo salvo em: {output_media}")
    except subprocess.CalledProcessError as e:
        print("\n❌ Ocorreu um erro ao executar o FaceFusion.")
        print("Lembre-se que o FaceFusion exige a instalação das dependências dele (`pip install -r facefusion/requirements.txt`)")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("❌ Uso Incorreto!\nSintaxe: python app/swap_faces.py caminho/rosto_A.jpg caminho/destino_B.jpg caminho/saida.jpg")
    else:
        swap_faces(sys.argv[1], sys.argv[2], sys.argv[3])
