import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(1)

# Modelo de mãos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

ultimo_tempo = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    # Desenhar mãos na tela
    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

    # CAPTURA A CADA 5 SEGUNDOS
    if time.time() - ultimo_tempo >= 5:
        ultimo_tempo = time.time()

        if resultado.multi_hand_landmarks:
            print("\n---- LOG (a cada 5s) ----")
            mao = resultado.multi_hand_landmarks[0]  # só a primeira mão
            for i, lm in enumerate(mao.landmark):
                print(f"({lm.x:.3f},{lm.y:.3f},{lm.z:.3f}),")
        else:
            print("\nNenhuma mão detectada no momento.")

    cv2.imshow("Gestos", frame)

    if cv2.waitKey(1) == 27:  # Tecla ESC para sair
        break

cap.release()
cv2.destroyAllWindows()