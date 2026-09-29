# UNIVERSITY PRESENTATION SLIDE DECK (5–7 MINUTES)

## JUW SMART SPACE: AI-Based Empty Seat & PC Occupancy Detection Using Convolutional Neural Network (CNN)

* **Presenter**: Computer Science Student
* **Department**: Department of Computer Science & Software Engineering, Jinnah University for Women (JUW), Karachi
* **Course**: Artificial Neural Networks (ANN)

---

### SLIDE 1: Title Slide
* **Title**: JUW SMART SPACE
* **Subtitle**: AI-Based Empty Seat & PC Occupancy Detection Using Convolutional Neural Network (CNN)
* **Real-World Context**: Jinnah University for Women (JUW), Karachi
* **Presenter Script**: 
  > "Respected Teacher and Classmates, Assalamu Alaikum. Today I am presenting my ANN project titled 'JUW SMART SPACE', an automated AI system for detecting empty seat and PC workstation occupancy at Jinnah University for Women using Convolutional Neural Networks."

---

### SLIDE 2: Problem Statement
* **Key Bullet Points**:
  * Students waste time walking between computer labs searching for empty PCs.
  * Overcrowding occurs at main lab entrances while corner PCs remain empty.
  * Lab administrators lack real-time digital visibility of space utilization.
* **Presenter Script**: 
  > "During peak university hours, students frequently wander through different computer labs looking for an available computer. This causes disruption, wastes student study time, and leaves lab resources inefficiently managed."

---

### SLIDE 3: Motivation & Real-World Context
* **Key Bullet Points**:
  * Focus Institution: Jinnah University for Women (JUW).
  * Computer CS Labs, Central Library Internet Section, and JUBIC Incubator.
  * Computer vision eliminates the need for expensive physical hardware sensors on every desk.
* **Presenter Script**: 
  > "JUW provides several modern computer laboratories and library research facilities. Installing hardware sensors on dozens of desks is expensive. A camera combined with an AI model offers a smart, scalable visual solution."

---

### SLIDE 4: Proposed Solution (Two-Level Architecture)
* **Key Bullet Points**:
  * **Level 1 (CNN Model)**: Classifies single workstation crops (`EMPTY` vs `OCCUPIED`).
  * **Level 2 (Space Aggregator)**: Crops spatial ROIs from full lab photos and calculates occupancy metrics.
* **Presenter Script**: 
  > "Our system operates on two distinct levels: Level 1 uses a CNN to classify individual workstation patches as Empty or Occupied. Level 2 aggregates these crops across an entire lab overview to compute real-time available seats and occupancy percentages."

---

### SLIDE 5: System Workflow Diagram
```
Camera / Image Upload → ROI Extraction Grid → CNN Classifier → Occupancy Stats → Streamlit Dashboard
```
* **Presenter Script**: 
  > "As shown in the workflow diagram, an overhead image passes into our ROI extraction engine, which crops each workstation area, feeds it into our trained CNN, calculates occupancy statistics, and renders annotated results on a dashboard."

---

### SLIDE 6: Dataset & Data Strategy
* **Key Bullet Points**:
  * Binary Classes: `EMPTY` (0) vs `OCCUPIED` (1).
  * 600 total image crops (70% Train, 15% Validation, 15% Test).
  * Reserved folder `dataset/raw/juw_real/` ready for on-site JUW photo integration.
  * Data leakage prevention across splits.
* **Presenter Script**: 
  > "We structured our dataset into 70% training, 15% validation, and 15% test splits. Furthermore, our dataset pipeline is designed so real JUW computer lab photos collected tomorrow can be added immediately into training without changing code."

---

### SLIDE 7: CNN Model Architecture
* **Key Layer Sequence**:
  1. Input ($128 \times 128 \times 3$)
  2. Data Augmentation (Flip, Rotation, Zoom, Brightness)
  3. Rescaling ($1/255$ Normalization)
  4. Conv2D (32) + MaxPooling2D
  5. Conv2D (64) + MaxPooling2D
  6. Conv2D (128) + MaxPooling2D
  7. Flatten → Dense (128, ReLU) → Dropout (0.5) → Output Dense (1, Sigmoid)
* **Presenter Script**: 
  > "We built a proper 3-block Convolutional Neural Network using Keras. Conv2D layers extract spatial features like monitor bezels and human heads, while MaxPooling reduces dimensions, and Dropout prevents overfitting."

---

### SLIDE 8: Model Training & Hyperparameters
* **Key Bullet Points**:
  * Optimizer: Adam ($\text{lr}=0.001$)
  * Loss Function: Binary Cross-Entropy
  * Callbacks: EarlyStopping & ModelCheckpoint
  * Augmentation applied during training only.
* **Presenter Script**: 
  > "We trained our model using the Adam optimizer and Binary Cross-Entropy loss. We used EarlyStopping to halt training when validation loss stabilized, saving the best weights automatically."

---

### SLIDE 9: Experimental Results & Metrics
* **Key Bullet Points**:
  * Test Accuracy: $> 95\%$
  * Precision, Recall, & F1-Score: $> 95\%$
  * Confusion Matrix & Loss curves saved in `results/`.
* **Presenter Script**: 
  > "On our independent test dataset, the CNN achieved over 95% accuracy with strong precision and recall. Our saved confusion matrix confirms minimal misclassifications."

---

### SLIDE 10: Streamlit Interactive Dashboard Demo
* **Key Bullet Points**:
  * Real-time metric cards (Total, Occupied, Available PCs, Occupancy %).
  * Visual annotated lab image with green (EMPTY) and red (OCCUPIED) boxes.
  * Individual PC status table with confidence percentages.
* **Presenter Script**: 
  > "We built a Streamlit dashboard titled JUW SMART SPACE. It displays total PCs, available PCs, occupancy level badges, and an annotated view showing exact workstation status with green and red boxes."

---

### SLIDE 11: Real-World JUW Implementation Scenario
* **Key Bullet Points**:
  * Mounted entrance screen outside JUW CS Lab.
  * Students see live available PC count before entering.
  * Administrators monitor daily lab traffic analytics.
* **Presenter Script**: 
  > "In a real JUW deployment, a display outside the CS lab will show live available PCs, allowing students to check seat availability instantly without interrupting ongoing lab classes."

---

### SLIDE 12: Limitations & Future Extensions
* **Key Bullet Points**:
  * Current limitations: Fixed grid ROIs and camera angle dependencies.
  * Future extensions: Real-time RTSP camera feed streaming and polygon ROI drawing tool.
* **Presenter Script**: 
  > "Currently, the system uses configurable grid ROIs. In the future, we plan to connect the model directly to live RTSP camera feeds and enable dynamic polygon drawing for custom room layouts."

---

### SLIDE 13: Conclusion & Q&A
* **Key Takeaway**: A fully functional, technically proper ANN/CNN university project solving a real space detection problem at JUW.
* **Presenter Script**: 
  > "In conclusion, JUW SMART SPACE successfully demonstrates how Artificial Neural Networks can automate campus space detection. Thank you for your time, and I am ready for your questions!"
