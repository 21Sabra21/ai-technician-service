"""
POST /api/analyze-problem
success  → top-1 ≥ 60%
possible → top-1 بين 40-59%
unknown  → top-1 < 40%
"""

import pickle, re, numpy as np, os
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nlp", "problem_analyzer.pkl")
_model_data = None

def _get_model():
    global _model_data
    if _model_data is None:
        with open(_MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)
    return _model_data


class AnalyzeProblemRequest(BaseModel):
    problemDescription: str = Field(..., min_length=1, max_length=500)

class RecommendedService(BaseModel):
    serviceId:   int
    serviceName: str
    confidence:  float

class AnalyzeProblemResponse(BaseModel):
    status: str
    recommendedServices: List[RecommendedService] = []
    message: str | None = None


_CONFIDENT  = 0.60
_POSSIBLE   = 0.40
_TOP2_MIN   = 0.30
_CLOSE_SURE = 0.35
_CLOSE_POSS = 0.20

_RULES = [
    # brakes
    (['فرامل','رج'],    'brakes'),  (['فرامل','هز'],    'brakes'),
    (['فرامل','رعش'],   'brakes'),  (['فرمل','رج'],     'brakes'),
    (['فرمل','هز'],     'brakes'),  (['فرمله','رج'],    'brakes'),
    (['احتكاك','فرامل'],'brakes'),  (['احتكاك','ببطء'], 'brakes'),
    (['صفار'],          'brakes'),  (['صفير'],          'brakes'),
    (['رجه','تقف'],     'brakes'),  (['رجه','فجا'],     'brakes'),
    (['رجه','فرمل'],    'brakes'),  (['رعشه','اشاره'],  'brakes'),
    (['احتكاك','تقف'],  'brakes'),  (['حكه','ببطء'],    'brakes'),
    # engine
    (['طف','هدي'],      'engine'),  (['طف','ارفع'],     'engine'),
    (['طف','اهدى'],     'engine'),  (['بطل','هدي'],     'engine'),
    (['بطل','ارفع'],    'engine'),  (['سحب','بنزين'],   'engine'),
    (['سحب','غاز'],     'engine'),  (['صوت','مطلع'],    'engine'),
    (['بيعلى','مطلع'],  'engine'),  (['تعب','مطلع'],    'engine'),
    (['تاخر','سرعه'],   'engine'),  (['تاخر','بنزين'],  'engine'),
    (['فصل','لحظه'],    'engine'),  (['فصل','ترجع'],    'engine'),
    (['تقطع','ماشي'],   'engine'),
    (['صوت','غريب','قليله'], 'engine'),
    (['بتمشي','بتقل'],  'engine'),  (['بتتحرك','بتقل'], 'engine'),
    (['بتتحرك','ببطء'], 'engine'),  (['ماشي','بتقل'],   'engine'),
    (['تقيله','مسافه'], 'engine'),  (['سلاسه','اقل'],   'engine'),
    (['ناعمه','سواقه'], 'engine'),  (['كركب'],          'engine'),
    (['اهتزاز','فجاه'], 'engine'),  (['صوتها','لكن','اعلي'], 'engine'),
    # exhaust
    (['ريح','غريب'],    'exhaust'), (['ريح','بنزين'],   'exhaust'),
    (['ريح','شياط'],    'exhaust'), (['ريح','حريق'],    'exhaust'),
    (['ريح','محروق'],   'exhaust'), (['ريح','عادم'],    'exhaust'),
    # tires
    (['طنين'],          'tires'),   (['صوت','زن'],      'tires'),
    (['زنه','مستمره'],  'tires'),   (['زنه','سرعه'],    'tires'),
    (['خبط','عجله'],    'tires'),
    # starting
    (['ساعات','بتدور'], 'starting'), (['ساعات','بتشتغل'], 'starting'),
    # alignment_balancing
    (['100','هز'], 'alignment_balancing'), (['100','رج'], 'alignment_balancing'),
    (['80','هز'],  'alignment_balancing'), (['80','رج'],  'alignment_balancing'),
    (['90','هز'],  'alignment_balancing'), (['120','هز'], 'alignment_balancing'),
    (['ثابته','طريق','سريع'],  'alignment_balancing'),
    (['ثابته','ملفات'],        'alignment_balancing'),
    (['رجرجه','هدي'],          'alignment_balancing'),
    (['رجرجه','اهدى'],         'alignment_balancing'),
    (['احتكاك','سرعه'],        'alignment_balancing'),
    # suspension
    (['طقطقه','الحركه'],      'suspension'),
    (['طقطقه','مع','الحركه'], 'suspension'),
    # cooling
    (['تسخن','اداء'],  'cooling'), (['صوت','يسخن'],  'cooling'),
    (['صوت','تسخن'],   'cooling'), (['تسخن','تعب'],   'cooling'),
]


def _normalize(text: str) -> str:
    for a, b in [('ة','ه'),('ى','ي'),('أ','ا'),('إ','ا'),('آ','ا'),('ئ','ي'),('ؤ','و')]:
        text = text.replace(a, b)
    return text

def _preprocess(text: str) -> str:
    text = _normalize(text.lower().strip())
    text = re.sub(r'[،,؟?!.]', ' ', text)
    return re.sub(r'\s+', ' ', text)

def _apply_rules(clean: str) -> list:
    seen = set()
    matched = []
    for kws, cat in _RULES:
        if cat not in seen and all(k in clean for k in kws):
            seen.add(cat)
            matched.append(cat)
    return matched

def _make(svc_map, cat, conf):
    s = svc_map[cat]
    return RecommendedService(serviceId=s["id"], serviceName=s["name"], confidence=round(conf, 3))


def _analyze(text: str) -> AnalyzeProblemResponse:
    data     = _get_model()
    pipeline = data["pipeline"]
    svc_map  = data["services"]
    clean    = _preprocess(text)

    # Model أولاً دايماً
    proba    = pipeline.predict_proba([clean])[0]
    classes  = pipeline.classes_
    idxs     = np.argsort(proba)[::-1]
    t1c, t1v = classes[idxs[0]], float(proba[idxs[0]])
    t2c, t2v = classes[idxs[1]], float(proba[idxs[1]])
    gap      = t1v - t2v

    # Rules: خدمات إضافية مختلفة عن الـ model
    rule_cats  = _apply_rules(clean)
    extra_svcs = [
        _make(svc_map, cat, 0.90)
        for cat in rule_cats
        if cat != t1c
    ]

    # Model مش واثق
    if t1v < _POSSIBLE or t1c == "unknown":
        if not extra_svcs:
            return AnalyzeProblemResponse(
                status="unknown",
                message="مش قدرنا نحدد المشكلة، ممكن توضح أكتر؟",
            )
        return AnalyzeProblemResponse(
            status="possible",
            recommendedServices=extra_svcs[:2],
            message="في احتمالين للمشكلة، التقني هيحدد بعد الفحص" if len(extra_svcs) > 1 else "مش متأكدين 100%، بس ده الأرجح",
        )

    # Model واثق — نبني svcs من الـ model
    if t1v >= _CONFIDENT:
        model_svcs = [_make(svc_map, t1c, t1v)]
        if gap <= _CLOSE_SURE and t2v >= _TOP2_MIN and t2c != "unknown":
            model_svcs.append(_make(svc_map, t2c, t2v))
    else:
        model_svcs = [_make(svc_map, t1c, t1v)]
        if gap <= _CLOSE_POSS and t2v >= _TOP2_MIN and t2c != "unknown":
            model_svcs.append(_make(svc_map, t2c, t2v))

    # ندمج extra_svcs بدون تكرار والعدد أقصاه 2
    seen_ids = {s.serviceId for s in model_svcs}
    for svc in extra_svcs:
        if svc.serviceId not in seen_ids and len(model_svcs) < 2:
            model_svcs.append(svc)
            seen_ids.add(svc.serviceId)

    if t1v >= _CONFIDENT:
        return AnalyzeProblemResponse(status="success", recommendedServices=model_svcs)

    msg = "في احتمالين للمشكلة، التقني هيحدد بعد الفحص" if len(model_svcs) > 1 \
          else "مش متأكدين 100%، بس ده الأرجح"
    return AnalyzeProblemResponse(status="possible", recommendedServices=model_svcs, message=msg)


router = APIRouter()

@router.post("/api/analyze-problem", response_model=AnalyzeProblemResponse)
async def analyze_problem(request: AnalyzeProblemRequest):
    return _analyze(request.problemDescription)