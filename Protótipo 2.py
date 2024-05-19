from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image  
import cv2
import torch
import numpy as np
import os

mtcnn = MTCNN()

# Inicializar o modelo InceptionResnetV1 pré-treinado
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Verificar se a GPU está disponível e configurar o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)

# Pasta contendo as fotos das pessoas conhecidas
known_faces_folder = 'C:\\Users\\lucia\\Desktop\\rostos'

# Carregar as imagens das pessoas conhecidas e suas respectivas embeddings faciais
known_faces = []
known_names = []
for filename in os.listdir(known_faces_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(known_faces_folder, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = mtcnn(img_pil, return_prob=False)  # Converter para tensor
        img_tensor = img_tensor.unsqueeze(0).to(device)
        embedding = resnet(img_tensor).detach().cpu().numpy()[0]
        known_faces.append(embedding)
        known_names.append(filename)

# Inicializar a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capturar um frame da câmera
    if not ret:
        break

    # Converter o frame para escala de cinza para detecção de faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos no frame usando MTCNN
    boxes, _ = mtcnn.detect(rgb_frame)

    # Desenhar retângulos ao redor dos rostos detectados e realizar reconhecimento facial
    if boxes is not None:
        for box in boxes:
            (x, y, w, h) = [int(coord) for coord in box]
            face = frame[y:h, x:w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Converter para RGB
            face_pil = Image.fromarray(face_rgb)
            face_tensor = mtcnn(face_pil, return_prob=False)  # Converter para tensor
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            
            # Comparar a embedding do rosto com as embeddings conhecidas
            recognized = False
            for i, known_embedding in enumerate(known_faces):
                distance = np.linalg.norm(embedding - known_embedding)
                if distance < 1.0:  # Defina um limite adequado para a distância
                    recognized = True
                    cv2.putText(frame, known_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    break
            
            # Desenhar retângulo ao redor do rosto detectado
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

    # Exibir o frame processado em uma janela
    cv2.imshow('Detecção e Reconhecimento de Rostos', frame)

    # Verificar se a tecla 'q' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos e fechar janelas ao sair do loop
cap.release()
cv2.destroyAllWindows()
