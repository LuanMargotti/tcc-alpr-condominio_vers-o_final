# Sistema ALPR Condominial (TCC)
Sistema de Reconhecimento Automático de Placas Veiculares focado em controle de acesso para condomínios, utilizando YOLOv8 e Visão Computacional.

## 📌 Comandos Essenciais do Git / GitHub

### 1. Verificar o que foi alterado (Opcional, mas recomendado)
Mostra quais arquivos foram modificados desde o último salvamento.
```
git status
```

### 2. Adicionar arquivos novos ao "Rascunho"
Adiciona todas as modificações feitas, respeitando as regras do .gitignore (ignorando pastas pesadas como env e datasets).
```
git add .
```

### 3. Criar o "Save Point" (Commit)
"Nomeia" as alterações que você acabou de adicionar, criando um marco na história do projeto.
```
git commit -m "Mensagem descritiva sobre as alterações"
```

Exemplo real:
```
git commit -m "Adicionado treino de caracteres com YOLOv8 e melhorias visuais no detector_final"
```

### 4. Enviar para a Nuvem (GitHub)
Envia as alterações salvas no seu computador para o repositório online.
```
git push origin main
```

## 🚀 Executando o Projeto

### Pré-requisitos
* Python 3.8 ou superior
* YOLOv8 (instalado via pip ou requirements.txt)
* OpenCV
* Pandas

### Treinamento
Para treinar o modelo com os dados disponibilizados:
```bash
python train.py
```

### Inferência/Detecção
Para testar o modelo em imagens ou vídeos:
```bash
python detector_final.py
```

## 📁 Estrutura do Projeto

* **models/**: Modelos YOLO pré-treinados e treinados.
* **datasets/**: Contém as imagens e anotações (se disponível).
* **utils/**: Utilitários e scripts de suporte (download de dados, checagem de GPU).
* **runs/**: Resultados de execução e métricas de treinamento.
