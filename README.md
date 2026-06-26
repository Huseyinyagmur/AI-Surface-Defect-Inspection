# AI Surface Defect Inspection

## Proje Hakkında

Bu proje, çelik yüzey kusurlarını derin öğrenme kullanarak otomatik olarak sınıflandırmak amacıyla geliştirilmiştir. Projede PyTorch tabanlı bir görüntü sınıflandırma sistemi oluşturulmuş ve Transfer Learning yaklaşımı kullanılarak ResNet18 modeli eğitilmiştir.

Proje, yalnızca model eğitmeye odaklanmak yerine profesyonel bir yapıda geliştirilmiştir. Veri hazırlama, model oluşturma, eğitim ve doğrulama işlemleri ayrı dosyalara ayrılarak modüler bir mimari oluşturulmuştur.

## Özellikler

* PyTorch tabanlı görüntü sınıflandırma sistemi
* Transfer Learning (ResNet18)
* GPU (CUDA) desteği
* Data Augmentation
* ImageFolder ve DataLoader kullanımı
* Eğitim ve doğrulama döngüsü
* En iyi modeli otomatik kaydetme (Model Checkpoint)
* Modüler proje yapısı

## Proje Yapısı

```text
AI-Surface-Defect-Inspection/
│
├── dataset/
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

## Kullanılan Teknolojiler

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib

## Eğitim Süreci

Model, ImageNet üzerinde önceden eğitilmiş ResNet18 mimarisi kullanılarak Transfer Learning yöntemiyle eğitilmektedir. Eğitim sırasında CrossEntropyLoss kayıp fonksiyonu ve Adam optimizasyon algoritması kullanılmaktadır. Validation doğruluğu her epoch sonunda hesaplanmakta ve en yüksek başarıya ulaşan model otomatik olarak kaydedilmektedir.

## Devam Eden Geliştirmeler

* Eğitim grafikleri (Loss & Accuracy)
* Confusion Matrix
* Classification Report
* Tek görüntü üzerinde tahmin (Inference)
* Model değerlendirme araçları
* README geliştirmeleri

## Amaç

Bu proje, bilgisayarlı görü ve derin öğrenme alanında profesyonel seviyede bir görüntü sınıflandırma sistemi geliştirmeyi ve PyTorch kullanımını gerçek bir proje üzerinde uygulamayı amaçlamaktadır.
