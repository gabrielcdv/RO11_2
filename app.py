# Streamlit app: upload a picture and compare the 3 methods
# (LBP + KNN, HOG + SVM, mini-Xception) on each face.
# Run with: streamlit run app.py

import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern
from torch import nn

# default paths (can be changed in the sidebar)
APP_DIR = Path(__file__).parent
DEFAULT_PATHS = {
    "LBP + KNN": APP_DIR / "models" / "modele_lbp_knn.joblib",
    "HOG + SVM": APP_DIR / "models" / "fer_hog_svm.joblib",
    "mini-Xception": APP_DIR / "models" / "mini_xception_fer2013.pt",
}

# order used for the display (same as the task: 0=Angry ... 6=Neutral)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# order of the CNN outputs = alphabetical order of the folders (ImageFolder)
CNN_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Feature extraction ----------------
# Same functions as in the notebooks: the features must be computed
# exactly like during training, otherwise the models give nonsense.

def caracteristiques_lbp(img, grille=4, P=8, R=1):
    # split the 48x48 face in grille x grille blocks
    h, w = img.shape
    taille_h, taille_w = h // grille, w // grille
    histogrammes = []
    for i in range(grille):
        for j in range(grille):
            bloc = img[i*taille_h:(i+1)*taille_h, j*taille_w:(j+1)*taille_w]
            # LBP code of each pixel (P neighbours on a circle of radius R)
            lbp_bloc = local_binary_pattern(bloc, P, R, method="default")
            # normalized 256-bin histogram of the codes
            hist, _ = np.histogram(lbp_bloc, bins=256, range=(0, 256))
            histogrammes.append(hist / hist.sum())
    # 16 blocks x 256 bins = vector of 4096 values
    return np.concatenate(histogrammes)


def image_to_hog(img):
    # histograms of gradient orientations on 8x8 cells, normalized by 2x2 blocks
    return hog(img.astype(np.float32), orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True, channel_axis=None)


# ---------------- mini-Xception ----------------
# Architecture copied from the CNN notebook: the .pt file only contains
# the weights, so the network has to be rebuilt before loading them.

class SeparableConv(nn.Sequential):
    # depthwise 3x3 conv (one filter per channel) then pointwise 1x1 conv
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                         nn.Conv2d(cin, cout, 1, bias=False))


class XceptionBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # main branch: 2 separable convs then max pooling (size divided by 2)
        self.main = nn.Sequential(
            SeparableConv(cin, cout), nn.BatchNorm2d(cout), nn.ReLU(),
            SeparableConv(cout, cout), nn.BatchNorm2d(cout),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # residual branch: 1x1 conv with stride 2 to get the same output size
        self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=2, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


def mini_xception(num_classes=7):
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, bias=False), nn.BatchNorm2d(8), nn.ReLU(),  # 48 -> 46
        nn.Conv2d(8, 8, 3, bias=False), nn.BatchNorm2d(8), nn.ReLU(),  # 46 -> 44
        XceptionBlock(8, 16), XceptionBlock(16, 32),                    # 44 -> 22 -> 11
        XceptionBlock(32, 64), XceptionBlock(64, 128),                  # 11 -> 6 -> 3
        nn.Conv2d(128, num_classes, 3, padding=1),                      # one map per class
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),                          # average -> 7 scores
    )


# ---------------- Model loading ----------------
# st.cache_resource: each model is loaded only once (and not at every click),
# useful because the KNN file is big (it stores all the training vectors).

@st.cache_resource
def load_lbp(path):
    # dict saved by the LBP notebook: {"knn", "grille", "P", "R"}
    return joblib.load(path)


@st.cache_resource
def load_hog(path):
    # dict saved by the HOG notebook: {"clf", "prob_clf"}
    return joblib.load(path)


@st.cache_resource
def load_cnn(path):
    model = mini_xception(len(CNN_CLASSES)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # inference mode (BatchNorm uses the saved statistics)
    with torch.no_grad():
        model(torch.zeros(1, 1, 48, 48, device=device))  # warm-up, otherwise the first timing is wrong
    return model


# ---------------- Prediction ----------------
# Each predict_xxx function takes a 48x48 grayscale face and returns
# (probabilities of the 7 emotions in EMOTIONS order, predicted emotion).

def reorder(classes, probs):
    # Each model has its own class order (ints for HOG, folder names for LBP and CNN),
    # so we put everything back in the EMOTIONS order.
    names = []
    for c in classes:
        if isinstance(c, (int, np.integer)):
            names.append(EMOTIONS[int(c)])
        else:
            names.append(str(c).capitalize())
    result = np.zeros(len(EMOTIONS))
    for name, p in zip(names, probs):
        result[EMOTIONS.index(name)] = p
    # argmax in the model order -> same tie-break as knn.predict()
    return result, names[np.argmax(probs)]


def predict_lbp(face, m):
    vecteur = caracteristiques_lbp(face, grille=m["grille"], P=m["P"], R=m["R"])
    # predict_proba of the KNN = share of the k neighbours in each class
    return reorder(m["knn"].classes_, m["knn"].predict_proba([vecteur])[0])


def predict_hog(face, m):
    clf = m["prob_clf"]  # calibrated version of the SVM, gives probabilities
    return reorder(clf.classes_, clf.predict_proba([image_to_hog(face)])[0])


def predict_cnn(face, model):
    # same normalization as in training: [0, 255] -> [-1, 1], shape (1, 1, 48, 48)
    x = torch.from_numpy(face).float().div(255).sub(0.5).div(0.5)[None, None].to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu().numpy()  # softmax: scores -> probabilities
    return reorder(CNN_CLASSES, probs)


# ---------------- Face detection ----------------
# Taken from the HOG notebook: OpenCV Haar cascades (Viola-Jones),
# pre-trained models shipped with opencv.

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_faces(gray, max_side=640):
    # detection on a reduced copy: faster and fewer false detections on big photos
    scale = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    # window +10 % at each scale, kept if 5 overlapping detections, min size 48 px
    boxes = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    # boxes (x, y, w, h) back to the size of the original image
    return [tuple(int(v / scale) for v in box) for box in boxes]


def align_face(gray, box, max_angle=30):
    x, y, w, h = box
    # look for the eyes in the upper half of the face only
    eyes = EYE_CASCADE.detectMultiScale(gray[y:y + h // 2, x:x + w], scaleFactor=1.1, minNeighbors=5)
    if len(eyes) >= 2:
        # keep the 2 biggest detections, sorted from left to right
        eyes = sorted(sorted(eyes, key=lambda e: e[2] * e[3])[-2:], key=lambda e: e[0])
        (lx, ly), (rx, ry) = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
        # angle of the line between the 2 eye centers
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))
        # ignore if the eyes are too close or the angle too big (probably a wrong detection)
        if rx - lx > w / 5 and abs(angle) <= max_angle:
            # rotation around the face center, shifted so that only the face box is rendered
            M = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), angle, 1.0)
            M[:, 2] -= (x, y)
            return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # eyes not found: simple crop without rotation
    return gray[y:y + h, x:x + w]


# ---------------- Interface ----------------

st.set_page_config(page_title="FER-2013 demo", layout="wide")
st.title("Facial emotion recognition (FER-2013)")

# sidebar: pre-processing options
st.sidebar.header("Options")
use_detection = st.sidebar.checkbox("Detect faces", value=True,
                                    help="Untick if the image is already a cropped face (e.g. from FER-2013)")
use_alignment = st.sidebar.checkbox("Align face on the eyes", value=True, disabled=not use_detection)

# sidebar: path of each model file
st.sidebar.header("Model files")
paths = {name: st.sidebar.text_input(name, str(p)) for name, p in DEFAULT_PATHS.items()}

# load the models that exist, a missing model only disables its method
methods = {}
for name, loader, predictor in [("LBP + KNN", load_lbp, predict_lbp),
                                ("HOG + SVM", load_hog, predict_hog),
                                ("mini-Xception", load_cnn, predict_cnn)]:
    if Path(paths[name]).is_file():
        methods[name] = (loader(paths[name]), predictor)
    else:
        st.sidebar.warning(f"{paths[name]} not found, {name} skipped")
if not methods:
    st.error("No model found, run the notebooks first.")
    st.stop()

# input image: file upload or webcam snapshot
tab1, tab2 = st.tabs(["Upload an image", "Webcam"])
uploaded = tab1.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"])
photo = tab2.camera_input("Take a photo")
source = photo or uploaded
if source is None:
    st.info("Upload an image or take a photo.")
    st.stop()  # nothing to do until an image is given

# exif_transpose: phone pictures are often stored rotated
rgb = np.array(ImageOps.exif_transpose(Image.open(source)).convert("RGB"))
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # the 3 models work in grayscale

# find the faces in the image
boxes = detect_faces(gray) if use_detection else []
if use_detection and not boxes:
    st.warning("No face detected, the whole image is used.")

# crop each face (aligned or not), or take the whole image if no face
if boxes:
    faces = []
    for (x, y, w, h) in boxes:
        faces.append(align_face(gray, (x, y, w, h)) if use_alignment else gray[y:y + h, x:x + w])
else:
    faces = [gray]

# draw the boxes with their number on a copy of the image
shown = rgb.copy()
thick = max(2, max(rgb.shape[:2]) // 300)  # line thickness adapted to the image size
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), thick)
    cv2.putText(shown, f"#{i + 1}", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, thick / 2, (0, 255, 0), thick)

# run the 3 methods on each face (resized to 48x48 like the dataset)
results = []
for face in faces:
    face48 = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    res = {}
    for name, (model, predictor) in methods.items():
        t = time.perf_counter()
        probs, label = predictor(face48, model)
        res[name] = (probs, label, (time.perf_counter() - t) * 1000)  # time in ms (features + model)
    results.append((face48, res))

# left: image with the boxes
left, right = st.columns(2)
left.image(shown, caption=f"{len(boxes)} face(s) detected", width="stretch")

# right: one line per face with the prediction of each method
summary = []
for i, (_, res) in enumerate(results):
    row = {"Face": f"#{i + 1}"}
    for name, (probs, label, _) in res.items():
        row[name] = f"{label} ({probs.max():.0%})"
    summary.append(row)
right.subheader("Summary")
right.dataframe(pd.DataFrame(summary).set_index("Face"), width="stretch")
# KNN with k=3 can only give 0, 33, 67 or 100 %
right.caption("Confidence: vote of the neighbours (KNN), calibrated probability (SVM), softmax (CNN)")

# details for each face
for i, (face48, res) in enumerate(results):
    st.divider()
    st.subheader(f"Face #{i + 1}")
    cols = st.columns([1, 2, 2, 2])
    # the 48x48 face really given to the models (enlarged without smoothing)
    cols[0].image(cv2.resize(face48, (144, 144), interpolation=cv2.INTER_NEAREST), caption="48x48 input")
    # predicted emotion, confidence and time for each method
    for col, (name, (probs, label, ms)) in zip(cols[1:], res.items()):
        col.metric(name, label)
        col.caption(f"Confidence {probs.max():.0%} | {ms:.1f} ms")

    # table of the 7 probabilities, shown as bars (one column per method)
    table = pd.DataFrame({name: r[0] * 100 for name, r in res.items()}, index=EMOTIONS)
    bars = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=100, format="%.0f%%")
            for name in table.columns}
    st.dataframe(table, column_config=bars, width="stretch")