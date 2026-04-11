"""
POST /api/analyze-problem
success  → top-1 ≥ 60%
possible → top-1 بين 40-59%
unknown  → top-1 < 40%

يرجع حتى MAX_SERVICES خدمات (model top-N + rules)
"""

import pickle, re, numpy as np, os, logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List

logger = logging.getLogger(__name__)

# ── Model path (robust للـ Railway وأي بيئة) ──────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "nlp", "problem_analyzer.pkl")
_model_data = None

def _get_model():
    global _model_data
    if _model_data is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {_MODEL_PATH}")
        with open(_MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)
        logger.info(f"✅ NLP model loaded from {_MODEL_PATH}")
    return _model_data


# ── Schemas ────────────────────────────────────────────────
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


# ── Thresholds ────────────────────────────────────────────
_CONFIDENT    = 0.60
_POSSIBLE     = 0.40
_TOP_N_MIN    = 0.25
_CLOSE_SURE   = 0.35
_CLOSE_POSS   = 0.20
_MAX_SERVICES = 3

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
    seen, matched = set(), []
    for kws, cat in _RULES:
        if cat not in seen and all(k in clean for k in kws):
            seen.add(cat)
            matched.append(cat)
    return matched

def _make(svc_map, cat, conf):
    s = svc_map[cat]
    return RecommendedService(serviceId=s["id"], serviceName=s["name"], confidence=round(conf, 3))

def _merge(base: list, extras: list, max_n: int) -> list:
    seen_ids = {s.serviceId for s in base}
    for svc in extras:
        if len(base) >= max_n:
            break
        if svc.serviceId not in seen_ids:
            base.append(svc)
            seen_ids.add(svc.serviceId)
    return base


def _analyze(text: str) -> AnalyzeProblemResponse:
    data     = _get_model()
    pipeline = data["pipeline"]
    svc_map  = data["services"]
    clean    = _preprocess(text)

    proba   = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    idxs    = np.argsort(proba)[::-1]

    t1c, t1v = classes[idxs[0]], float(proba[idxs[0]])
    t2c, t2v = classes[idxs[1]], float(proba[idxs[1]])
    t3c, t3v = classes[idxs[2]], float(proba[idxs[2]])
    gap12 = t1v - t2v
    gap23 = t2v - t3v

    logger.debug(f"NLP → t1={t1c}({t1v:.2f}) t2={t2c}({t2v:.2f}) t3={t3c}({t3v:.2f})")

    rule_cats  = _apply_rules(clean)
    extra_svcs = [
        _make(svc_map, cat, 0.90)
        for cat in rule_cats
        if cat != t1c
    ]

    # Unknown zone
    if t1v < _POSSIBLE or t1c == "unknown":
        if not extra_svcs:
            return AnalyzeProblemResponse(
                status="unknown",
                message="مش قدرنا نحدد المشكلة، ممكن توضح أكتر؟",
            )
        result = extra_svcs[:_MAX_SERVICES]
        msg = (
            "في أكتر من احتمال للمشكلة، التقني هيحدد بعد الفحص"
            if len(result) > 1 else "مش متأكدين 100%، بس ده الأرجح"
        )
        return AnalyzeProblemResponse(status="possible", recommendedServices=result, message=msg)

    # Model confident
    model_svcs = [_make(svc_map, t1c, t1v)]

    if t1v >= _CONFIDENT:
        if gap12 <= _CLOSE_SURE and t2v >= _TOP_N_MIN and t2c != "unknown":
            model_svcs.append(_make(svc_map, t2c, t2v))
            if gap23 <= _CLOSE_SURE and t3v >= _TOP_N_MIN and t3c != "unknown":
                model_svcs.append(_make(svc_map, t3c, t3v))
    else:
        if gap12 <= _CLOSE_POSS and t2v >= _TOP_N_MIN and t2c != "unknown":
            model_svcs.append(_make(svc_map, t2c, t2v))
            if gap23 <= _CLOSE_POSS and t3v >= _TOP_N_MIN and t3c != "unknown":
                model_svcs.append(_make(svc_map, t3c, t3v))

    model_svcs = _merge(model_svcs, extra_svcs, _MAX_SERVICES)

    if t1v >= _CONFIDENT:
        return AnalyzeProblemResponse(status="success", recommendedServices=model_svcs)

    msg = (
        "في أكتر من احتمال للمشكلة، التقني هيحدد بعد الفحص"
        if len(model_svcs) > 1 else "مش متأكدين 100%، بس ده الأرجح"
    )
    return AnalyzeProblemResponse(status="possible", recommendedServices=model_svcs, message=msg)


# ── Router ────────────────────────────────────────────────
router = APIRouter()

@router.post("/api/analyze-problem", response_model=AnalyzeProblemResponse)
async def analyze_problem(request: AnalyzeProblemRequest):
    try:
        return _analyze(request.problemDescription)
    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        return JSONResponse(
            status_code=503,
            content={"error": "NLP model not loaded", "detail": str(e)}
        )
    except Exception as e:
        logger.error(f"analyze-problem error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "فيه مشكلة في التحليل، حاول تاني"}
        )