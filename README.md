# Emotion Detection Using Facial Expressions

This project is a real-time facial emotion detection system using computer vision and deep learning. It captures facial expressions from a webcam feed, processes the face region, and classifies the emotion into one of seven categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

---

## 🚀 **Features**
- **Real-time Emotion Detection**: Uses your webcam to detect and classify facial emotions live.
- **Deep Learning Model**: Built using a Convolutional Neural Network (CNN) trained on emotion datasets.
- **User-Friendly Visualization**: Displays bounding boxes around detected faces along with the predicted emotion label.

---

## 🛠️ **Technologies Used**
1. **Programming Language**: Python
2. **Libraries**:
   - OpenCV: For video processing and face detection
   - Keras: For loading and utilizing the deep learning model
   - NumPy: For numerical computations
3. **Pre-Trained Components**:
   - Haar Cascade Classifier: For face detection
   - Trained Emotion Classification Model (`model.h5`): For emotion recognition

---

## 📋 **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install opencv-python keras numpy
```

### 3. Place Required Files
- Download or ensure the presence of:
  - `haarcascade_frontalface_default.xml`: The Haar Cascade face detector file.
  - `model.h5`: Pre-trained deep learning model for emotion classification.
- Place these files in the same directory as the script.

### 4. Run the Program
Start the webcam-based emotion detection:
```bash
python emotion_detector.py
```

---

## 🖼️ **How It Works**
1. **Face Detection**: Detects faces in the webcam feed using Haar Cascade Classifier.
2. **Preprocessing**:
   - Converts the detected face region to grayscale.
   - Resizes the face to 48x48 pixels for compatibility with the deep learning model.
3. **Emotion Prediction**:
   - Normalizes the pixel values to [0,1].
   - Predicts the emotion using the trained CNN model.
   - Displays the emotion label on the live feed.

---

## 📂 **Project Files**
- `emotion_detector.py`: Main script for running the emotion detection system.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade model for face detection.
- `model.h5`: Trained CNN model for emotion classification.

---

## 🎯 **Usage**
- **Start the program** and allow access to your webcam.
- The system will:
  - Detect your face.
  - Predict and display your current emotion in real-time.

---

## 🔧 **Customization**
- **Improve the Model**: Train your custom model on a different or larger emotion dataset.
- **Fine-tune Detection**: Adjust parameters in the `detectMultiScale` method for better face detection accuracy.
- **Additional Features**: Integrate this system with other applications like mood-based music players or chatbots.

---

## 🤝 **Contributions**
Contributions are welcome! If you have any improvements or suggestions:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## 📜 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 **Contact**
For any questions or feedback, feel free to contact me at:
- **Email**: yourname@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

---

## 📝 **Acknowledgments**
- **OpenCV**: For providing powerful image processing tools.
- **Keras**: For simplifying deep learning model implementation.
- **Haar Cascade**: For reliable face detection. 

Happy coding! 😊
