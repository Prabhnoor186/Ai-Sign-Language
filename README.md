# AI Sign Language Recognition

A real-time American Sign Language (ASL) alphabet recognizer. It detects a hand from the webcam, preprocesses the hand region, and classifies the sign with a trained Keras model, displaying the predicted letter live on screen.

**Supported signs:** A–P (16 letters, static hand poses)

## How It Works

1. **Hand detection:** `cvzone`'s `HandDetector` (built on MediaPipe) finds the hand and returns its bounding box.
2. **Preprocessing:** the hand region is cropped with a margin, resized to fit a 300×300 white canvas while preserving its aspect ratio, so every sample has a consistent shape.
3. **Data collection:** `datacollection.py` saves the preprocessed image to a per-letter folder under `data/` each time you press `s`.
4. **Classification:** the saved images were used to train a Keras model **[CONFIRM: trained with Google Teachable Machine / custom Keras code]**, exported as `model/keras_model.h5` with its labels in `model/labels.txt`.
5. **Real-time prediction:** `test.py` runs the same preprocessing on each webcam frame, passes it to the model, and draws the predicted letter with a bounding box on the live feed.

## Tech Stack

- Python
- OpenCV
- cvzone (hand tracking and classification wrapper)
- MediaPipe
- TensorFlow / Keras
- NumPy

## Project Structure

```
Ai-Sign-Language/
├── data/               # Collected training images, one folder per letter
├── model/
│   ├── keras_model.h5  # Trained classifier
│   └── labels.txt      # Class labels
├── datacollection.py   # Captures and saves training images from the webcam
└── test.py             # Real-time sign prediction
```

## Installation

```bash
git clone https://github.com/Prabhnoor186/Ai-Sign-Language.git
cd Ai-Sign-Language
pip install opencv-python cvzone mediapipe numpy tensorflow
```

A webcam is required.

## Usage

**Collect training data**

1. Set `folder` in `datacollection.py` to the letter you're recording, e.g. `data/A`.
2. Run:
   ```bash
   python datacollection.py
   ```
3. Hold the sign in front of the camera and press `s` to save each image. The terminal prints how many you've saved.

**Run real-time recognition**

```bash
python test.py
```

Show a sign to the camera. The predicted letter appears above your hand. Press `Ctrl+C` in the terminal to stop.



## Limitations

- Recognizes static poses only. Letters that involve motion (e.g. J) are not captured by a single-frame model.
- Works best with one hand, good lighting and a plain background.
- Limited to A–P in the current version.

## Future Improvements

- Extend to the full A–Z alphabet and common words
- Handle motion-based signs with a sequence model
- Combine predicted letters into words and sentences
- Package as a web or mobile app

## Author

**Prabhnoor Kaur**
[GitHub](https://github.com/Prabhnoor186) | [LinkedIn](https://www.linkedin.com/in/prabhnoor-kaur-78b23b)
