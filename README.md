# FER-2013 emotion demo

Small Streamlit app to test our 3 models on any picture:
- LBP + KNN
- HOG + linear SVM
- mini-Xception (CNN)

Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Installation

```
pip install -r requirements-app.txt
```

The app does not train anything, it loads the models saved by the notebooks.
Run the 3 notebooks first, then put the saved models next to the app:

```
app.py
models/modele_lbp_knn.joblib
models/fer_hog_svm.joblib
models/mini_xception_fer2013.pt
```

Other locations can be given in the sidebar ("Model files").
If one model is missing, the app still works with the other methods.

## Usage

```
streamlit run app.py
```

Then open http://localhost:8501 in a browser and:
1. upload an image (jpg, png, webp...) or take a photo with the webcam tab,
2. each detected face is shown with its number,
3. for each face you get the prediction of the 3 methods, the confidence, the inference time
   and the probabilities for the 7 emotions.

## Options (sidebar)

- **Detect faces**: finds the faces with an OpenCV Haar cascade, crops them and resizes them to 48x48
  (the models were trained on cropped 48x48 faces). Untick it for images already cropped,
  like the ones from the FER-2013 dataset.
- **Align face on the eyes**: rotates the face so the eyes are horizontal before classification.
- **Model files**: path of each saved model.

## Notes

- The Haar cascade only finds faces seen roughly from the front. If no face is found,
  the whole image is used.
- With k = 3 the KNN confidence can only be 0, 33, 67 or 100 %.
- The first run is a bit slow because the KNN model (which stores all the training features) is big.