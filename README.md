# AI Surface Defect Inspection

## 📌 Proje Hakkında

Bu proje, çelik yüzeylerde oluşan kusurların **Derin Öğrenme (Deep Learning)** teknikleri kullanılarak otomatik olarak sınıflandırılması amacıyla geliştirilmiştir.

Proje kapsamında **PyTorch** kütüphanesi kullanılarak görüntü sınıflandırma sistemi oluşturulmuş, **Transfer Learning** yaklaşımı ile **ResNet18** mimarisi yeniden eğitilmiştir.

Model geliştirme sürecinde yalnızca eğitim aşamasına odaklanılmamış; veri hazırlama, model oluşturma, eğitim, tahmin (Inference) ve model değerlendirme süreçleri birbirinden ayrılarak modüler ve sürdürülebilir bir proje mimarisi oluşturulmuştur.

---

# 🚀 Özellikler

* PyTorch tabanlı görüntü sınıflandırma sistemi
* Transfer Learning (ResNet18)
* GPU (CUDA) desteği
* Data Augmentation
* ImageFolder ve DataLoader kullanımı
* Eğitim (Training) ve doğrulama (Validation) süreçleri
* En başarılı modeli otomatik kaydetme (Model Checkpoint)
* Tek görüntü üzerinde tahmin (Inference)
* Confusion Matrix
* Classification Report
* Precision, Recall ve F1-Score değerlendirmeleri
* Eğitim Loss grafiği
* Validation Accuracy grafiği
* Modüler proje yapısı

---

# 📂 Proje Yapısı

```text
AI-Surface-Defect-Inspection/
│
├── dataset/
├── models/
│   └── best_model.pth
│
├── results/
│   └── plots/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Kullanılan Teknolojiler

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Scikit-learn
* Pillow

---

# 🧠 Model Mimarisi

Projede **ImageNet** veri kümesi üzerinde önceden eğitilmiş **ResNet18** mimarisi kullanılmıştır.

Transfer Learning yaklaşımı sayesinde modelin son tam bağlantılı (Fully Connected) katmanı, veri kümesindeki sınıf sayısına uygun olacak şekilde yeniden düzenlenmiş ve yalnızca sınıflandırma katmanı yeniden eğitilmiştir.

**Model Yapılandırması**

* Model: ResNet18
* Framework: PyTorch
* Transfer Learning
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* GPU (CUDA) desteği

---

# ⚙️ Eğitim Süreci

Model eğitimi aşağıdaki adımlardan oluşmaktadır:

1. Veri setinin yüklenmesi
2. Data Augmentation uygulanması
3. ResNet18 modelinin oluşturulması
4. Transfer Learning uygulanması
5. Eğitim (Training)
6. Doğrulama (Validation)
7. En başarılı modelin kaydedilmesi
8. Eğitim sonuçlarının görselleştirilmesi

Eğitim sonunda otomatik olarak;

* Training Loss Grafiği
* Validation Accuracy Grafiği

oluşturulmaktadır.

---

# 🔍 Model Değerlendirmesi

Eğitim tamamlandıktan sonra model, ayrı bir değerlendirme modülü kullanılarak analiz edilmektedir.

Değerlendirme kapsamında;

* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1-Score

hesaplanmaktadır.

---

# 🖼️ Tahmin (Inference)

Eğitilmiş model kullanılarak tek bir görüntü üzerinde tahmin yapılabilmektedir.

Tahmin sürecinde;

* Görüntünün yüklenmesi
* Ön işleme (Preprocessing)
* Model tahmini
* Softmax olasılık hesaplaması
* Tahmin edilen sınıfın belirlenmesi
* Güven skorunun hesaplanması

işlemleri gerçekleştirilmektedir.

Örnek çıktı:

```text
Prediction : crazing
Confidence : 99.93%
```

---

# 📊 Sonuçlar

Model, doğrulama veri kümesi üzerinde yüksek sınıflandırma başarısı elde etmiştir.

Validation veri kümesi toplam **360 görüntüden** oluşmaktadır. Bu veri kümesinde yüksek doğruluk elde edilmiş olsa da modelin gerçek dünya performansının daha kapsamlı değerlendirilebilmesi için daha büyük ve çeşitli veri kümeleri üzerinde test edilmesi önerilmektedir.

---

# 🎯 Projenin Amacı

Bu proje aşağıdaki konularda uygulamalı deneyim kazanmak amacıyla geliştirilmiştir:

* Derin Öğrenme
* Bilgisayarlı Görü (Computer Vision)
* Transfer Learning
* Görüntü Sınıflandırma
* PyTorch
* Model Değerlendirme
* Profesyonel AI Proje Mimarisi

---

# 🚀 Gelecekte Yapılabilecek Geliştirmeler

* TensorBoard entegrasyonu
* ONNX model dönüşümü
* TensorRT optimizasyonu
* Grad-CAM görselleştirmesi
* Hyperparameter Optimization
* EfficientNet ve ConvNeXt gibi farklı mimarilerin eklenmesi

---

# 👨‍💻 Geliştirici

Bu proje, PyTorch kullanılarak Bilgisayarlı Görü (Computer Vision) ve Derin Öğrenme alanında uygulamalı deneyim kazanmak amacıyla geliştirilmiştir.
