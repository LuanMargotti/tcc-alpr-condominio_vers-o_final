import cv2
from ultralytics import YOLO

# 1. Configurações
modelo = YOLO('runs/detect/meu_tcc_alpr/treino_caracteres/weights/best.pt')
# Simulando a "Lista Branca" do condomínio
lista_branca = ["BMP4D29", "ABC1D23", "LUA1234"] 

caminho_imagem = 'Placas-Brasil-2/valid/images/FQP9171_png_jpg.rf.994d5d46129f27c9079356a965027dc3.jpg'

# 2. Executa a Inteligência Artificial
resultados = modelo(caminho_imagem)

# 3. Lógica para extrair o texto da placa
# O YOLO detecta caracteres soltos, vamos ordená-los da esquerda para a direita
deteccoes = []
for box in resultados[0].boxes:
    x_centro = box.xywh[0][0].item() # Posição horizontal para ordenar
    classe_id = int(box.cls[0].item())
    caractere = modelo.names[classe_id]
    deteccoes.append((x_centro, caractere))

# Ordena pelo X (esquerda para a direita) e junta as letras
deteccoes.sort()
placa_detectada = "".join([d[1] for d in deteccoes]).upper()

print(f"\n" + "="*30)
print(f"PLACA IDENTIFICADA: {placa_detectada}")

# 4. Verificação de Segurança (Coração do TCC)
if placa_detectada in lista_branca:
    status = "ACESSO LIBERADO"
    cor = (0, 255, 0) # Verde
else:
    status = "ACESSO BLOQUEADO"
    cor = (0, 0, 255) # Vermelho

print(f"STATUS: {status}")
print("="*30)

# 5. Visualização
img = resultados[0].plot()
cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)
cv2.imshow("Sistema ALPR Condominial", img)
cv2.waitKey(0)
cv2.destroyAllWindows()