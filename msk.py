"""
نماذج معالجة الصور وفهمها
Image Processing & Understanding Models

يستعيد هذا الكود نماذج مبنية مسبقاً لفهم الصور وتصنيفها
"""

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO


# ─────────────────────────────────────────
# 1. تحميل نموذج من torchvision
# ─────────────────────────────────────────

def load_torchvision_model(model_name: str = "resnet50", pretrained: bool = True):
    """
    يستعيد نموذج معالجة صور من torchvision
    النماذج المتاحة: resnet50, efficientnet_b0, vit_b_16, densenet121, ...
    """
    print(f"\n📦 تحميل نموذج: {model_name}")
    
    model_map = {
        "resnet50":        models.resnet50,
        "resnet18":        models.resnet18,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b4": models.efficientnet_b4,
        "densenet121":     models.densenet121,
        "vit_b_16":        models.vit_b_16,
        "mobilenet_v3":    models.mobilenet_v3_large,
        "convnext_small":  models.convnext_small,
    }

    if model_name not in model_map:
        raise ValueError(f"النموذج '{model_name}' غير متاح. الخيارات: {list(model_map.keys())}")

    weights_param = "DEFAULT" if pretrained else None
    model = model_map[model_name](weights=weights_param)
    model.eval()  # وضع التقييم

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ تم تحميل النموذج | المعاملات: {total_params:,}")
    return model


# ─────────────────────────────────────────
# 2. معالجة وتجهيز الصورة
# ─────────────────────────────────────────

def preprocess_image(image_source, image_size: int = 224):
    """
    تجهيز الصورة للنماذج: تغيير الحجم، التطبيع، التحويل لـ tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225]      # ImageNet std
        ),
    ])

    if isinstance(image_source, str) and image_source.startswith("http"):
        response = requests.get(image_source, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    elif isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise TypeError("image_source يجب أن يكون رابط URL أو مسار ملف أو PIL.Image")

    tensor = transform(img).unsqueeze(0)  # إضافة بُعد الـ batch
    print(f"  🖼  حجم الصورة بعد المعالجة: {tensor.shape}")
    return tensor, img


# ─────────────────────────────────────────
# 3. تصنيف الصورة مع أفضل K نتائج
# ─────────────────────────────────────────

def classify_image(model, image_tensor, top_k: int = 5):
    """
    يُصنّف الصورة ويعيد أفضل top_k نتائج مع الثقة
    """
    # تحميل أسماء تصنيفات ImageNet
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        labels = requests.get(labels_url, timeout=10).json()
    except Exception:
        labels = [f"class_{i}" for i in range(1000)]

    with torch.no_grad():
        output = model(image_tensor)           # التنبؤ
        probs = torch.softmax(output, dim=1)   # تحويل إلى احتمالات

    top_probs, top_indices = probs.topk(top_k, dim=1)

    print(f"\n🏆 أفضل {top_k} نتائج:")
    results = []
    for rank, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0])):
        label = labels[idx.item()] if idx.item() < len(labels) else f"class_{idx.item()}"
        confidence = prob.item() * 100
        print(f"  {rank+1}. {label:30s} → {confidence:.2f}%")
        results.append({"rank": rank + 1, "label": label, "confidence": confidence})

    return results


# ─────────────────────────────────────────
# 4. استخراج الميزات (Feature Extraction)
# ─────────────────────────────────────────

def extract_features(model, image_tensor, model_name: str = "resnet50"):
    """
    يستخرج ميزات الصورة من الطبقة قبل الأخيرة (feature vector)
    مفيد للـ similarity search أو fine-tuning
    """
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.flatten(1)

    print(f"\n🔬 ميزات الصورة (Feature Vector):")
    print(f"  الشكل: {features.shape}  |  المتوسط: {features.mean().item():.4f}  |  الانحراف: {features.std().item():.4f}")
    return features


# ─────────────────────────────────────────
# 5. Hugging Face Transformers (Vision Models)
# ─────────────────────────────────────────

def load_huggingface_vit(model_id: str = "google/vit-base-patch16-224"):
    """
    يستعيد نموذج Vision Transformer من Hugging Face
    """
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        print(f"\n🤗 تحميل نموذج Hugging Face: {model_id}")
        processor = ViTImageProcessor.from_pretrained(model_id)
        model = ViTForImageClassification.from_pretrained(model_id)
        model.eval()
        print(f"  ✅ النموذج جاهز | التصنيفات: {model.config.num_labels}")
        return model, processor
    except ImportError:
        print("  ⚠️  ثبّت المكتبة: pip install transformers")
        return None, None


def classify_with_hf(model, processor, image_source, top_k: int = 5):
    """
    تصنيف الصورة باستخدام Hugging Face Vision Transformer
    """
    if model is None:
        return []

    if isinstance(image_source, str) and image_source.startswith("http"):
        response = requests.get(image_source, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(top_k, dim=1)

    print(f"\n🏆 أفضل {top_k} نتائج (ViT HuggingFace):")
    results = []
    for rank, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0])):
        label = model.config.id2label.get(idx.item(), f"class_{idx.item()}")
        confidence = prob.item() * 100
        print(f"  {rank+1}. {label:40s} → {confidence:.2f}%")
        results.append({"rank": rank + 1, "label": label, "confidence": confidence})

    return results


# ─────────────────────────────────────────
# 6. عرض معلومات النموذج
# ─────────────────────────────────────────

def model_info(model):
    """يطبع ملخصاً لبنية النموذج"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 معلومات النموذج:")
    print(f"  إجمالي المعاملات  : {total:>15,}")
    print(f"  معاملات قابلة للتدريب: {trainable:>15,}")
    print(f"  الجهاز المستخدم   : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────
# 🚀 التشغيل الرئيسي
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ── صورة تجريبية من الإنترنت ──
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"

    print("=" * 55)
    print("   نماذج معالجة الصور وفهمها  🖼️ 🤖")
    print("=" * 55)

    # 1️⃣ تحميل نموذج ResNet-50
    model = load_torchvision_model("resnet50", pretrained=True)
    model_info(model)

    # 2️⃣ تجهيز الصورة
    print("\n⚙️  معالجة الصورة...")
    tensor, pil_img = preprocess_image(IMAGE_URL)

    # 3️⃣ تصنيف الصورة
    results = classify_image(model, tensor, top_k=5)

    # 4️⃣ استخراج الميزات
    features = extract_features(model, tensor)

    # 5️⃣ تجربة Vision Transformer من HuggingFace (اختياري)
    hf_model, processor = load_huggingface_vit("google/vit-base-patch16-224")
    if hf_model:
        classify_with_hf(hf_model, processor, IMAGE_URL, top_k=5)

    print("\n✅ انتهى التنفيذ بنجاح!")
