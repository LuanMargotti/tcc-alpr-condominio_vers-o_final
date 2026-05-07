from ultralytics import YOLO
import torch
import os

def treinar():
    # O caminho real que vimos no seu print:
    caminho_last = 'runs/detect/meu_tcc_alpr/treino_caracteres/weights/last.pt'
    
    if os.path.exists(caminho_last):
        print(f"Retomando treinamento do checkpoint: {caminho_last}")
        # Carregamos o modelo direto do arquivo last.pt
        modelo = YOLO(caminho_last)
        
        if torch.cuda.is_available():
            print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")

        # Iniciamos o treino com resume=True
        modelo.train(
            data='Placas-Brasil-2/data.yaml', 
            epochs=50, 
            imgsz=640, 
            device=0, 
            workers=0, # Mantemos 0 para estabilidade no Windows
            resume=True       
        )
    else:
        print(f"ERRO: Arquivo não encontrado em: {caminho_last}")
        print("Verifique se o nome das pastas no Explorer está exatamente igual.")

if __name__ == '__main__':
    treinar()