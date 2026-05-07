import cv2
import glob

# 1. Caminhos (ajuste se a sua pasta de imagens estiver em outro lugar)
pasta_imagens = 'datasets/Placas-Brasil-2/valid/images/*.jpg'
caminho_video_saida = 'datasets/video_teste.mp4'

# Busca todas as imagens na pasta
lista_imagens = glob.glob(pasta_imagens)

if not lista_imagens:
    print("Nenhuma imagem encontrada. Verifique o caminho da pasta!")
else:
    print(f"Encontradas {len(lista_imagens)} imagens. Montando o vídeo...")

    # Pega o tamanho da primeira imagem para configurar a tela do vídeo
    frame_inicial = cv2.imread(lista_imagens[0])
    altura, largura, _ = frame_inicial.shape

    # Configura o gravador de vídeo (FPS = 2, para dar tempo de você ver a detecção)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(caminho_video_saida, fourcc, 2.0, (largura, altura))

    # Costura as imagens no vídeo
    for img_path in lista_imagens[:50]: # Pega as 50 primeiras imagens
        img = cv2.imread(img_path)
        img_redimensionada = cv2.resize(img, (largura, altura))
        video.write(img_redimensionada)

    video.release()
    print(f"Sucesso! Vídeo salvo em: {caminho_video_saida}")