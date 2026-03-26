import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('car_problems_dataset_v6.csv')
print(f"✅ Loaded {len(df)} records")
print(df['category'].value_counts())

def normalize(text):
    """Arabic normalization - unify similar characters"""
    text = str(text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')
    text = text.replace('ئ', 'ي')
    text = text.replace('ؤ', 'و')
    return text

def preprocess(text):
    text = normalize(text)
    text = text.lower().strip()
    text = re.sub(r'[،,؟?!.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_text'] = df['description'].apply(preprocess)
X = df['clean_text'].values
y = df['category'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=25000,
            sublinear_tf=True
        )),
        ('word', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True
        )),
    ])),
    ('clf', CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=5
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

cv = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"📊 Cross-validation: {cv.mean():.2%} ± {cv.std():.2%}")

# Confusion Matrix
classes = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            ax=ax, linewidths=0.5)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix - v6 Model (16 Categories)', fontsize=15, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('confusion_matrix_v6.png', dpi=150, bbox_inches='tight')
print("✅ Saved confusion_matrix_v6.png")

# Print errors
print("\n❌ الأخطاء:")
errors = [(X_test[i], y_test[i], y_pred[i])
          for i in range(len(y_test)) if y_test[i] != y_pred[i]]
for text, true, pred in errors:
    print(f"  '{text}'\n   True: {true} → Pred: {pred}\n")
print(f"إجمالي الأخطاء: {len(errors)} من {len(y_test)}")

# ── Services Map ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
SERVICES_MAP = {
    "oil_change":           {"id": 1,  "name": "تغيير زيت المحرك"},
    "oil_leak":             {"id": 2,  "name": "إصلاح تسريب الزيت"},
    "brakes":               {"id": 3,  "name": "إصلاح الفرامل"},
    "tires":                {"id": 4,  "name": "تغيير إطارات"},
    "alignment_balancing":  {"id": 5,  "name": "ضبط الزوايا والتوازن"},
    "suspension":           {"id": 6,  "name": "إصلاح نظام التعليق"},
    "steering":             {"id": 7,  "name": "إصلاح نظام التوجيه"},
    "transmission":         {"id": 8,  "name": "إصلاح ناقل الحركة"},
    "ac":                   {"id": 9,  "name": "فحص وإصلاح المكيف"},
    "cooling":              {"id": 10, "name": "إصلاح نظام التبريد"},
    "engine":               {"id": 11, "name": "فحص المحرك الإلكتروني"},
    "exhaust":              {"id": 12, "name": "إصلاح نظام العادم"},
    "starting":             {"id": 13, "name": "إصلاح مشاكل التشغيل"},
    "electrical":           {"id": 14, "name": "إصلاح الكهرباء"},
    "battery":              {"id": 15, "name": "إصلاح البطارية"},
    "cleaning":             {"id": 16, "name": "تنظيف وتلميع السيارة"},
    "unknown":              {"id": 17, "name": "غير ذلك"},
}

def predict(text):
    clean = preprocess(text)
    proba = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    best_idx = np.argmax(proba)
    best_cat = classes[best_idx]
    confidence = float(proba[best_idx])
    if confidence < CONFIDENCE_THRESHOLD or best_cat == 'unknown':
        return {"status": "unknown", "confidence": round(confidence, 3)}
    svc = SERVICES_MAP[best_cat]
    return {
        "status": "success",
        "category": best_cat,
        "confidence": round(confidence, 3),
        "recommendedService": {"serviceId": svc["id"], "serviceName": svc["name"]}
    }

# print("\n🧪 اختبار شامل:")
# print("=" * 70)
# examples = [
#     ("العربية بترعش وهي واقفة ودايس فرامل",  "engine"),
#     ("العربية بتتهز لما اشغلها وهي باردة",    "engine"),
#     ("العربية بترج لما امشي بسرعة",            "alignment_balancing"),
#     ("لقيت بقعة زيت تحت العربية",             "oil_leak"),
#     ("العربية بتبطل لما اشغل التكييف",         "engine"),
#     ("في زنة مستمرة من العجل",                 "tires"),
#     ("الشكمان بيطلع دخان أسود",               "exhaust"),
#     ("في دخان أبيض من الشكمان",               "exhaust"),
#     ("العربية مش بتشتغل من أول مرة",          "starting"),
#     ("العربية بتطفى مني فجأة",                "engine"),
#     ("البطارية فضت",                           "battery"),
#     ("البطارية بتفضى بسرعة",                  "battery"),
#     ("الدركسيون تقيل لما بلف",                "steering"),
#     ("في صوت طقطقة لما الف الدركسيون",        "steering"),
#     ("العربية بتهز بعد الثمانين",              "alignment_balancing"),
#     ("محتاج أعمل ضبط زوايا",                  "alignment_balancing"),
#     ("العربية بتسخن في الزحمة",               "cooling"),
#     ("لقيت مية نازلة تحت العربية",            "cooling"),
#     ("الفرامل بتصرصر",                        "brakes"),
#     ("التكييف مش بيبرد",                      "ac"),
#     ("الكاوتش اتفش",                          "tires"),
#     ("الفتيس بينقل متأخر",                    "transmission"),
#     ("عايز أغسل العربية",                     "cleaning"),
#     ("موعد تغيير الزيت جه",                   "oil_change"),
#     ("مرحبا",                                 "unknown"),
#     ("طريقة عمل البامية",                     "unknown"),
#     ("كام السعر",                              "unknown"),
#     # الجمل الجديدة
#     ("العربية مش مرتاحة في السحب",            "engine"),
#     ("العربية فيها تزييق جامد",               "steering"),
#     ("العربية بترزع لما اطلع مطب",            "suspension"),
#     ("العربية بتفرفر وأنا ماشي",              "engine"),
#     ("في ريحة محروقة من العربية",             "exhaust"),
#     ("العربية مكتومة ومش بتسحب",              "engine"),
# ]

# correct = 0
# for text, expected in examples:
#     result = predict(text)
#     ok = (result['status'] == 'success' and result.get('category') == expected) or \
#          (result['status'] == 'unknown' and expected == 'unknown')
#     if ok: correct += 1
#     icon = "✅" if ok else "❌"
#     if result['status'] == 'success':
#         print(f"{icon} '{text}'\n   → {result['category']} | {result['confidence']:.0%}\n")
#     else:
#         print(f"{icon} '{text}'\n   → unknown | {result['confidence']:.0%}\n")

# print(f"النتيجة: {correct}/{len(examples)} صح")

# Save
model_data = {
    'pipeline':              pipeline,
    'services':              SERVICES_MAP,
    'version':               '6.0',
    'accuracy':              round(accuracy, 4),
    'confidence_threshold':  CONFIDENCE_THRESHOLD,
    'model_type':            'TF-IDF (char+word) + SVM',
    'language':              'arabic_egyptian',
    'categories':            list(SERVICES_MAP.keys()),
}

with open('models/nlp/problem_analyzer.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved | Accuracy: {accuracy:.2%}")