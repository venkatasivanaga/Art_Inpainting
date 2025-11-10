# 🎨 Prior-Guided Art Inpainting with Local Edge-Tuned Texture Enhancement

## 🧠 Overview
This project restores damaged artworks — such as scratched, cracked, or missing regions — while preserving the original artistic style, brushwork, and texture.  
It combines **prior-guided inpainting** with **edge-tuned local enhancement** to produce curator-friendly restorations that maintain authenticity and transparency.

The system learns from both damaged and undamaged artworks and reconstructs missing areas with style-consistent textures.  
A lightweight UI allows users to upload an image, define a mask (manually or automatically), and preview restorations with change heatmaps before exporting results.

---

## 📁 Project Structure
```
.
├─ src/
│  └─ setup.ipynb             # Contains the python scripts
├─ notes/
│  ├─ docx/
├── Report                # contains the report related to project
└── README.md              # README.md
```

---

## 🧩 Problem Context
Museums and archives face challenges with digitized artworks suffering from cracks, tears, or missing regions.  
Manual digital retouching is time-consuming and subjective.  

This project provides an AI-based restoration pipeline that:
- Preserves **style consistency** (brushwork, color harmony, surface grain)
- Maintains **sharp boundaries** using edge-aware guidance
- Offers **transparent restorations** via heatmaps and watermarked exports

---

## 🖼️ Datasets

### 1. Damaged & Undamaged Artworks (Kaggle)
- **Source:** [Kaggle - Damaged and Undamaged Artworks](https://www.kaggle.com/datasets/pes1ug22am047/damaged-and-undamaged-artworks)
- **Usage:** Primary dataset for classifier training and inpainting experiments  
- **Download:**
  ```bash
  kaggle datasets download -d pes1ug22am047/damaged-and-undamaged-artworks -p data/kaggle_art_damage --unzip
  ```

### 2. Art Images — Clear and Distorted (Kaggle)
- **Source:** [Kaggle - Art Images Clear and Distorted](https://www.kaggle.com/datasets/sankarmechengg/art-images-clear-and-distorted)
- **Usage:** Auxiliary dataset for self-supervised inpainting and robustness testing  
- **Download:**
  ```bash
  kaggle datasets download -d sankarmechengg/art-images-clear-and-distorted -p data/kaggle_art_clear_distorted --unzip
  ```

---

## 🧮 Frameworks & Libraries
- **PyTorch**, **TorchVision**, **PyTorch Lightning**
- **OpenCV**, **scikit-image**, **NumPy**, **Matplotlib**
- **LPIPS**, **pytorch-fid** for perceptual metrics
- **Streamlit** or **Gradio** for UI


---

## 🚀 How to Run

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/venkatasivanaga/Art_Inpainting.git
cd Art_Inpainting
```

### 2. Download Datasets
```bash
Run the dataset download commands above or execute `setup.ipynb`.
```
---

## 🗓️ Implementation Timeline

| Week | Focus | Outcome |
|------|--------|----------|
| Oct 20–26 | Data pipeline & mask generation | Dataset ready, EDA complete |
| Oct 27–Nov 2 | Baseline classifier + basic UI | Functional classifier & mock UI |
| Nov 3–16 | Core inpainting model | Prior-guided restoration working |
| Nov 17–30 | Seam sharpening + UX | Interactive demo (Streamlit) |
| Dec 1–11 | Final polish | Presentation-ready results |

---

## 👩‍💻 Author
**Venkata Siva Reddy Naga**  
_Data Science | vs.naga@ufl.edu 

