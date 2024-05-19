from facenet_pytorch import MTCNN
import cv2

# Inicializar o detector MTCNN
mtcnn = MTCNN()

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

    # Desenhar retângulos ao redor dos rostos detectados
    if boxes is not None:
        for box in boxes:
            (x, y, w, h) = [int(coord) for coord in box]
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    # Exibir o frame processado em uma janela
    cv2.imshow('Detecção de Rostos', frame)

    # Verificar se a tecla 'q' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos e fechar janelas ao sair do loop
cap.release()
cv2.destroyAllWindows()
