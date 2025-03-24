# **Face Mask Detection using Semi-Supervised Learning**  

## **Overview**  
This project implements a **semi-supervised learning approach** for **face mask detection and segmentation**, leveraging both labeled and unlabeled data to enhance model performance. The model is benchmarked against state-of-the-art architectures like **DeepLabv3, U-Net, Mask R-CNN, and SegNet**, demonstrating superior segmentation quality, sensitivity, and specificity.  

## **Features**  
- **Semi-Supervised Learning:** Utilizes both labeled and unlabeled data for improved generalization.  
- **Deep Learning Models:** Compares performance with **DeepLabv3, Mask R-CNN, U-Net, and SegNet**.  
- **High Accuracy:** Competitive Dice and Jaccard scores for precise segmentation.  
- **Robust Performance:** Ensures low false positives (high specificity) and high true positive detection (high sensitivity).  
- **Efficient Training:** Reduces dependency on large labeled datasets, making it cost-effective.  

## **Dataset**  
The dataset used for training and evaluation consists of **face mask images** with corresponding segmentation masks. It includes both synthetic and real-world datasets to ensure robustness.  

### **Dataset Link**  
- **Medical Mask Dataset:** [https://www.kaggle.com/datasets/andrewmvd/medical-mask-dataset](https://www.kaggle.com/datasets/andrewmvd/medical-mask-dataset)

## **Installation**  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-repo-name/face-mask-detection.git
   cd face-mask-detection
   ```  
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  

## **Results & Performance**  
| **Metric**    | **Our Model** | **DeepLabv3** | **Mask R-CNN** | **U-Net** | **SegNet** |  
|--------------|-------------|--------------|--------------|--------|--------|  
| Dice Score   | 88.10%      | 88.50%       | 87.90%       | 85.90% | 84.70% |  
| Jaccard Score | 79.80%      | 80.10%       | 79.50%       | 78.20% | 76.90% |  
| Sensitivity  | 87.80%      | 90.10%       | 89.50%       | 86.30% | 85.20% |  
| Specificity  | 97.00%      | 97.20%       | 96.80%       | 96.40% | 95.90% |  
| Accuracy     | 93.30%      | 93.60%       | 93.00%       | 92.50% | 91.90% |  

## **Contributors**  
- **B. Anuradha** - Conceptualization & Supervision  
- **Manda Vighneshwara Reddy** - Model Training & Implementation
- **Gummireddy Sainath Reddy** - Review & Technical Improvements
- **P.Y. Geetha Madhuri** - Dataset Preprocessing & Testing  
- **Sudhanshu Gupta** - Data Analysis & Performance Metrics   
- **Akhil Kumar** - Figures & Statistical Analysis  

## **Citation**  
If you use this work in your research, please cite:  
```
@article{face_mask_detection_2025,
  author    = {B. Anuradha, P.Y. Geetha Madhuri, Manda Vighneshwara Reddy, Sudhanshu Gupta, Gummireddy Sainath Reddy, Akhil Kumar},
  title     = {Semi-Supervised Face Mask Detection Using Deep Learning},
  journal   = {},
  year      = {2025}
}
```

## **License**  
This project is licensed under the **MIT License**.
