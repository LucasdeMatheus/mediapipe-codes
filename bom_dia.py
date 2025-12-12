import cv2
import mediapipe as mp
import time
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# FUNÇÃO: NORMALIZAR LANDMARKS
# ==============================
def normalizar_landmarks(landmarks):
    lm = np.array(landmarks, dtype=float)

    # 1) Centraliza no ponto 0 (pulso)
    base = lm[0]
    lm -= base

    # 2) Escala para tamanho padrão
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm /= max_val

    return lm.tolist()


# ==============================
# SUAS POSES ORIGINAIS
# ==============================

pose_ultimo_noite = [
    (0.255,0.474,0.000),(0.283,0.444,-0.000),(0.305,0.405,0.000),(0.325,0.385,-0.002),
    (0.340,0.374,-0.003),(0.272,0.369,0.014),(0.297,0.358,0.005),(0.320,0.359,-0.005),
    (0.340,0.363,-0.011),(0.260,0.366,0.009),(0.295,0.355,-0.001),(0.321,0.357,-0.010),
    (0.344,0.360,-0.015),(0.251,0.367,0.002),(0.290,0.359,-0.009),(0.314,0.362,-0.016),
    (0.335,0.366,-0.017),(0.244,0.374,-0.006),(0.279,0.368,-0.015),(0.300,0.371,-0.019),
    (0.318,0.372,-0.019)
]

pose_primeiro_noite = [
   (0.198,0.487,0.000),(0.228,0.477,-0.009),(0.258,0.447,-0.011),(0.285,0.431,-0.014),
   (0.307,0.433,-0.017),(0.255,0.391,0.002),(0.269,0.356,-0.005),(0.278,0.334,-0.012),
   (0.287,0.314,-0.018),(0.242,0.384,-0.000),(0.258,0.346,-0.006),(0.269,0.322,-0.012),
   (0.280,0.300,-0.017),(0.227,0.383,-0.005),(0.243,0.345,-0.012),(0.254,0.323,-0.016),
   (0.264,0.303,-0.019),(0.209,0.388,-0.010),(0.223,0.359,-0.017),(0.234,0.342,-0.019),
   (0.244,0.326,-0.020)
]

pose_ultimo_dia = [
    (0.310,0.564,0.000),
(0.301,0.510,0.006),
(0.311,0.461,0.004),
(0.336,0.440,-0.001),
(0.360,0.441,-0.006),
(0.304,0.441,-0.002),
(0.325,0.388,-0.007),
(0.340,0.362,-0.009),
(0.351,0.343,-0.011),
(0.323,0.451,-0.010),
(0.358,0.410,-0.018),
(0.362,0.430,-0.018),
(0.353,0.446,-0.017),
(0.344,0.467,-0.018),
(0.373,0.427,-0.022),
(0.374,0.446,-0.015),
(0.365,0.459,-0.010),
(0.362,0.485,-0.025),
(0.384,0.451,-0.023),
(0.385,0.461,-0.014),
(0.377,0.472,-0.007)
]

pose_primeiro_dia = [
    (0.388,0.613,-0.000),
(0.380,0.554,0.010),
(0.393,0.510,0.010),
(0.414,0.495,0.007),
(0.435,0.501,0.004),
(0.396,0.486,-0.000),
(0.410,0.436,-0.002),
(0.423,0.414,-0.002),
(0.433,0.400,-0.002),
(0.418,0.498,-0.009),
(0.447,0.462,-0.012),
(0.446,0.480,-0.009),
(0.438,0.495,-0.006),
(0.438,0.517,-0.016),
(0.462,0.484,-0.016),
(0.458,0.499,-0.008),
(0.449,0.511,-0.003),
(0.454,0.539,-0.022),
(0.469,0.507,-0.016),
(0.465,0.518,-0.005),
(0.458,0.529,0.003)
]

pose_ultimo_bom = [
    (0.383,0.597,0.000),
(0.360,0.543,-0.006),
(0.349,0.481,-0.015),
(0.344,0.430,-0.020),
(0.338,0.392,-0.027),
(0.373,0.473,-0.042),
(0.372,0.395,-0.064),
(0.375,0.346,-0.076),
(0.378,0.307,-0.083),
(0.399,0.488,-0.046),
(0.427,0.410,-0.065),
(0.447,0.361,-0.074),(0.460,0.323,-0.080),(0.423,0.505,-0.049),
(0.455,0.437,-0.067),(0.474,0.394,-0.075),(0.488,0.359,-0.080),(0.443,0.523,-0.050),
(0.481,0.481,-0.065),(0.502,0.459,-0.071),(0.519,0.437,-0.073)
]

pose_primeiro_bom = [
    (0.361,0.610,0.000),(0.351,0.555,0.007),(0.360,0.507,0.005),(0.383,0.485,0.001),
    (0.405,0.482,-0.003),(0.360,0.486,-0.005),(0.385,0.438,-0.009),(0.402,0.413,-0.011),
    (0.412,0.393,-0.012),(0.380,0.497,-0.013),(0.415,0.456,-0.019),(0.419,0.473,-0.017),
    (0.410,0.488,-0.015),(0.401,0.514,-0.020),(0.432,0.477,-0.023),(0.431,0.495,-0.015),
    (0.421,0.508,-0.009),(0.419,0.535,-0.027),(0.443,0.500,-0.024),(0.442,0.512,-0.014),
    (0.433,0.524,-0.007),
]

pose_oi = [
(0.401,0.584,0.000),(0.427,0.563,-0.004),(0.446,0.535,-0.004),(0.459,0.510,-0.006),
(0.459,0.485,-0.008),(0.408,0.497,0.015),(0.428,0.471,0.008),(0.445,0.473,0.001),
(0.456,0.480,-0.002),(0.402,0.492,0.011),(0.423,0.463,0.002),(0.441,0.467,-0.007),
(0.453,0.477,-0.011),(0.396,0.489,0.005),(0.420,0.463,-0.006),(0.439,0.467,-0.012),
(0.452,0.476,-0.013),(0.392,0.488,-0.002),(0.401,0.451,-0.006),(0.405,0.427,-0.007),
(0.407,0.410,-0.007)
]

# ==============================
# NORMALIZA TODAS AS POSES
# ==============================
pose_oi = normalizar_landmarks(pose_oi)
pose_primeiro_bom = normalizar_landmarks(pose_primeiro_bom)
pose_ultimo_bom = normalizar_landmarks(pose_ultimo_bom)
pose_primeiro_dia = normalizar_landmarks(pose_primeiro_dia)
pose_ultimo_dia = normalizar_landmarks(pose_ultimo_dia)
pose_primeiro_noite = normalizar_landmarks(pose_primeiro_noite)
pose_ultimo_noite = normalizar_landmarks(pose_ultimo_noite)


# ==============================
# FUNÇÕES DE COMPARAÇÃO
# ==============================
def distancia_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )

def landmarks_proximos(lm1, lm2, tol=0.32):
    lm1 = normalizar_landmarks(lm1)
    dist = [distancia_3d(a, b) for a, b in zip(lm1, lm2)]
    return max(dist) <= tol


# ==============================
# ESTADO DAS SEQUÊNCIAS
# ==============================
estado = {
    "pose_oi": False,
    "primeiro_bom": False,
    "ultimo_bom": False,
    "primeiro_dia": False,
    "ultimo_dia": False,
    "primeiro_noite": False,
    "ultimo_noite": False
}

# ==============================
# LOOP DA CÂMERA
# ==============================
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            current = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            #mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ==============================
        # DETECÇÃO DOS GESTOS
        # ==============================

        if not estado["pose_oi"] and landmarks_proximos(current, pose_oi):
            estado["pose_oi"] = True
            print("\n===========\n  OI! \n===========\n")

        if estado["pose_oi"] and not estado["primeiro_bom"] and landmarks_proximos(current, pose_primeiro_bom):
            estado["primeiro_bom"] = True
            print("→ Primeiro BOM")

        if estado["primeiro_bom"] and not estado["ultimo_bom"] and landmarks_proximos(current, pose_ultimo_bom):
            estado["ultimo_bom"] = True
            print("→ Último BOM")

        if estado["ultimo_bom"] and not estado["primeiro_dia"] and landmarks_proximos(current, pose_primeiro_dia):
            estado["primeiro_dia"] = True
            print("→ Primeiro DIA")

        if estado["primeiro_dia"] and not estado["ultimo_dia"] and landmarks_proximos(current, pose_ultimo_dia):
            print("\n===========\n  BOM DIA! \n===========\n")
            estado = {k: False for k in estado}

        if estado["ultimo_bom"] and not estado["primeiro_noite"] and landmarks_proximos(current, pose_primeiro_noite):
            estado["primeiro_noite"] = True
            print("→ Primeiro NOITE")

        if estado["primeiro_noite"] and not estado["ultimo_noite"] and landmarks_proximos(current, pose_ultimo_noite):
            print("\n===========\n  BOA NOITE! \n===========\n")
            estado = {k: False for k in estado}

    cv2.imshow("Gestos", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()