# Beyond Accuracy: Comparison of ResNet50 and GWN-Enhanced Models for Brain Tumor MRI Classification with LIME Visualization

Official implementation of the paper *"Beyond Accuracy: Comparison of ResNet50 and GWN-Enhanced Models for Brain Tumor MRI Classification with LIME visualization"*.

## Authors

**Nguyen Thanh Minh Tam¹**, Mai Nhu Yen¹, Nguyen Quang Huy¹, Nguyen Thi Nhung¹, Nguyen Thi Huyen Chau¹, Nguyen Hoang Phuong¹, Dong Van He², Bui Xuan Cuong², and Vladik Kreinovich³

¹ *Faculty of Information Technology, Thang Long University, Hanoi, Vietnam*  
² *Center for Neurosurgery, Viet Duc Hospital, Hanoi, Vietnam*  
³ *Department of Computer Science, University of Texas at El Paso, USA*  

---

## Abstract

Recent studies report remarkably high accuracy (>99%) for brain tumor MRI classification using deep convolutional neural networks. However, the reliability of such high accuracy is questionable for public slice-based MRI datasets due to slice-level data leakage. 

In this study, we conduct a visualizable comparative analysis between a baseline ResNet50 model and an enhanced version incorporating Ghost Weight Normalization (GWN) across 3,300 brain MRI images (No Tumor, Glioma, Pituitary). Even though both models achieve high metrics (~0.99 Accuracy, F1-score, Macro-AUC), training dynamics show that **ResNet50+GWN converges faster (4–5 epochs vs 7–8 epochs) and produces smoother loss trajectories**. LIME-based visual explanations reveal that GWN induces a more spatially compact attention pattern compared to vanilla ResNet50.

---

## Key Features

- **ResNet50 + GWN Architecture**: Ghost Weight Normalization recursively applied to convolutional and linear layers without altering backbone structure.
- **Controlled Dataset Setup**: 3,300 balanced images (1,100 per class: `notumor`, `glioma`, `pituitary`) split 80/10/10.
- **Two-Phase Fine-Tuning**:
  - *Phase 1 (Warm-up)*: Frozen backbone, learning rate `1e-4` (5 epochs).
  - *Phase 2 (Fine-tuning)*: Full network unfrozen, learning rate `5e-5` with `ReduceLROnPlateau` scheduler.
- **LIME Explainability**: Superpixel-based local explanations to audit model decision sensitivity and background reliance.

---

## Dataset Structure

Download the public Kaggle Brain Tumor MRI dataset and organize it as follows:

```text
Training/
├── glioma/
├── notumor/
└── pituitary/
```

---

## Requirements & Setup

```bash
git clone https://github.com/minhtamngynt-art/brain-tumor-gwn-lime.git
cd brain-tumor-gwn-lime
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
```text
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
pillow
matplotlib
scikit-learn
scikit-image
lime
tqdm
```

---

## Results

### Performance Summary

| Model | Accuracy | Macro AUC | F1-Score | Optimal Convergence Epoch |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet50 Baseline** | 99.9% | 0.99 | 0.99 | ~23 |
| **ResNet50 + GWN** | 99.9% | 0.99 | 0.99 | **~10** |

### Per-Class Evaluation

| Class | Model | Accuracy | AUC | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **No Tumor** | ResNet50 / ResNet50+GWN | 99.9% | 0.99 | 0.99 |
| **Glioma** | ResNet50 / ResNet50+GWN | 99.9% | 0.99 | 0.99 |
| **Pituitary** | ResNet50 / ResNet50+GWN | 99.9% | 0.99 | 0.99 |

---

## Usage

### 1. Jupyter Notebook

Run the complete pipeline (preprocessing, model building, two-phase training, evaluation, and LIME visualization) in:

```bash
jupyter notebook notebooks/brain_tumor_resnet50_gwn_lime.ipynb
```

---

## Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@inproceedings{minhtam2026beyond,
  title={Beyond Accuracy: Comparison of ResNet50 and GWN-Enhanced Models for Brain Tumor MRI Classification with LIME visualization},
  author={Nguyen Thanh Minh Tam and Mai Nhu Yen and Nguyen Quang Huy and Nguyen Thi Nhung and Nguyen Thi Huyen Chau and Nguyen Hoang Phuong and Dong Van He and Bui Xuan Cuong and Vladik Kreinovich},
  booktitle={Proceedings of International Conference on Information Technology and Intelligent Systems (ICTIS)},
  year={2026}
}
```
