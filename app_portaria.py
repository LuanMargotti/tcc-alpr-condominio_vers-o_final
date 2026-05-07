import cv2
from ultralytics import YOLO

def iniciar_sistema():
    print("Carregando o 'cérebro' do ALPR...")
    # Carrega o modelo treinado
    modelo = YOLO('runs/detect/meu_tcc_alpr/treino_caracteres/weights/best.pt')

    # Simulando a "Lista Branca" do condomínio
    lista_branca = ["BMP4D29", "ABC1D23", "LUA1234"]

    # Aponta para o vídeo de teste
    caminho_video = 'datasets/video_teste.mp4'
    cap = cv2.VideoCapture(caminho_video) 

    print("="*40)
    print("SISTEMA ALPR INICIADO")
    print("CONTROLES:")
    print("[ESPAÇO] ou [ENTER] -> Avança para a próxima placa")
    print("[ Q ] -> Encerra o sistema")
    print("="*40)

    # Configuração da janela para forçar a TELA CHEIA
    nome_janela = "Sistema ALPR Condominial - TCC Luan"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        sucesso, frame = cap.read()
        
        # Se o vídeo acabar, rebobina para o começo (loop infinito)
        if not sucesso:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Executa a Inteligência Artificial no frame atual
        resultados = modelo(frame, stream=True, verbose=False) 

        for r in resultados:
            # Cria uma cópia da imagem com os retângulos da detecção
            frame_anotado = r.plot() 
            
            deteccoes = []
            # Lê cada caractere e pega as posições X e Y
            for box in r.boxes:
                # xywh = [centro_x, centro_y, largura, altura]
                x_centro, y_centro, w, h = box.xywh[0].tolist() 
                classe_id = int(box.cls[0].item())
                caractere = modelo.names[classe_id]
                deteccoes.append({'x': x_centro, 'y': y_centro, 'h': h, 'char': caractere})
            
            # Se a IA encontrou letras na tela
            if len(deteccoes) > 0:
                # Verifica a diferença de altura (Eixo Y) para saber se é moto
                min_y = min(d['y'] for d in deteccoes)
                max_y = max(d['y'] for d in deteccoes)
                media_altura = sum(d['h'] for d in deteccoes) / len(deteccoes)

                # Regra: Se a distância entre o topo e o fundo for maior que metade da altura da letra... É MOTO!
                if (max_y - min_y) > (media_altura * 0.5):
                    # Descobre onde é o "meio" da placa no eixo Y
                    media_y = sum(d['y'] for d in deteccoes) / len(deteccoes)
                    
                    # Separa quem tá na linha de cima e quem tá na linha de baixo
                    linha_cima = [d for d in deteccoes if d['y'] < media_y]
                    linha_baixo = [d for d in deteccoes if d['y'] >= media_y]
                    
                    # Ordena cada linha da esquerda para a direita (Eixo X)
                    linha_cima = sorted(linha_cima, key=lambda k: k['x'])
                    linha_baixo = sorted(linha_baixo, key=lambda k: k['x'])
                    
                    # Junta a de cima primeiro, e a de baixo depois
                    placa_detectada = "".join([d['char'] for d in linha_cima]) + "".join([d['char'] for d in linha_baixo])
                else:
                    # É CARRO! (Tudo na mesma linha). Ordena só pelo eixo X normalmente
                    deteccoes = sorted(deteccoes, key=lambda k: k['x'])
                    placa_detectada = "".join([d['char'] for d in deteccoes])
                
                placa_detectada = placa_detectada.upper()
                
                # Validação de Segurança
                if placa_detectada in lista_branca:
                    status = f"LIBERADO: {placa_detectada}"
                    cor = (0, 255, 0) # Verde
                else:
                    status = f"BLOQUEADO: {placa_detectada}"
                    cor = (0, 0, 255) # Vermelho
                
                # Amplia a imagem (Zoom de 4x)
                altura, largura = frame_anotado.shape[:2]
                frame_grande = cv2.resize(frame_anotado, (largura * 4, altura * 4))

                # Cria uma borda preta de 100 pixels no topo do vídeo para o painel de texto
                frame_grande = cv2.copyMakeBorder(frame_grande, 100, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

                # Escreve o status na tarja preta
                cv2.putText(frame_grande, status, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 3)

                # Exibe a janela em tela cheia
                cv2.imshow(nome_janela, frame_grande)

                # FREIO MANUAL: Pausa até você apertar espaço
                tecla = cv2.waitKey(0) & 0xFF
                
                # Se a tecla for 'q', desliga a câmera e fecha o programa
                if tecla == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return 

    # Liberação de segurança
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    iniciar_sistema()