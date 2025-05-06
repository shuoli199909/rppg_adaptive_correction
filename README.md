# Adaptive Physiologically Constrained Signal Correction Improves Contactless Heart Rate Monitoring


---

## 🔧 Project Structure

```
rppg_adaptive_correction/
│
├── arduino/                   # Arduino test for different methods
│   └── anduino_test.ino
│
├── config/                    # Configuration files
├── data/                      # Raw and processed input data
│
├── filters/                   # Filtering & signal processing utilities
│   ├── index_filter.py
│   ├── kalman_filter.py
│   ├── moving_average.py
│   ├── outlier_detection.py
│   └── peak_verification.py
│
├── main/                      # Entry points for various processing stages
│   ├── main_bvpwin2HR.py
│   ├── main_gen_gtHR.py
│   ├── main_rgb2bvpwin.py
│   └── main_vid2rgb.py
│
├── processor/                 # HR post processing & index process
│   ├── index_processor.py
│   └── post_processor.py
│
├── result/                    # Output data and evaluation results
├── util/                      # Utility functions
│
├── .gitignore
├── LICENSE
└── README.md
```

---


🧠 Key Features
- 💡 Multiple rPPG Algorithms: Supports initial heart rate estimation using various rPPG methods, enabling flexible input signal processing.
- 📊 Post-Processing Comparison: Evaluates and compares the accuracy of different heart rate post-processing algorithms (e.g., Kalman Filter, Moving Average, Index Filter).
- 🔌 Arduino Deployment Benchmarking: Analyzes the computational performance of post-processing methods when deployed on Arduino, helping assess real-time feasibility on edge devices.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Pipeline

1. **Extract RGB Traces from Video**:
   ```bash
   python main/main_vid2rgb.py
   ```

2. **Convert RGB to BVP Window**:
   ```bash
   python main/main_rgb2bvpwin.py
   ```

3. **Estimate Heart Rate**:
   ```bash
   python main/main_bvpwin2HR.py
   ```

4. *(Optional)* Generate Ground Truth HR for Comparison:
   ```bash
   python main/main_gen_gtHR.py
   ```

---

## 📊 Results

Results will be saved in the `result/` directory. The processed HR estimations can be visualized and compared against ground truth values for accuracy evaluation.

---

## 🤝 Contributing

We welcome pull requests and improvements from the community. Please ensure your code is well-commented and tested.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## ✨ Acknowledgments

Thanks to all contributors and the open-source community that inspired this work.

---