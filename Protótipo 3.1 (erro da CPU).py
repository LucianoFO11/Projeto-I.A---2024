from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import numpy as np
import os

print('Iniciando...') # Coloquei só pra ter certeza que tava rodando

# Inicia o MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
print(device)

# Inicia o modelo ResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
        img_tensor = mtcnn(img_rgb)  # Converter para tensor
        if img_tensor is not None and len(img_tensor) > 0:
            img_tensor = img_tensor[0].to(device)  # Seleciona o primeiro tensor da lista
            embedding = resnet(img_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
            known_faces.append(embedding)
            known_names.append(filename.split('.')[0])

# Inicializar a captura de vídeo da câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

while True:
    ret, frame = cap.read()  # Capturar um frame da câmera
    if not ret:
        break

    # Converter o frame para RGB para detecção de faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos no frame usando MTCNN
    try:
        boxes, _ = mtcnn.detect(rgb_frame)

        # Verificar se algum rosto foi detectado
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                (x1, y1, x2, y2) = [int(coord) for coord in box]
                face = frame[y1:y2, x1:x2]
                if face.size != 0:  # Verificar se o rosto não está vazio
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Converter para RGB
                    face_tensor = mtcnn(face_rgb)  # Converter para tensor
                    if face_tensor is not None and len(face_tensor) > 0:
                        face_tensor = face_tensor[0].to(device)  # Seleciona o primeiro tensor da lista
                        embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]

                        # Comparar a embedding do rosto com as embeddings conhecidas
                        recognized = False
                        for i, known_embedding in enumerate(known_faces):
                            distance = np.linalg.norm(embedding - known_embedding)
                            if distance < 1.0:  # Defina um limite adequado para a distância
                                recognized = True
                                cv2.putText(frame, known_names[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                                break

                        # Desenhar retângulo ao redor do rosto detectado
                        color = (0, 255, 0) if recognized else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    except Exception as e:
        pass

    # Exibir o frame processado em uma janela
    cv2.imshow('Detecção e Reconhecimento de Rostos', frame)

    # Verificar se a tecla 'q' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
