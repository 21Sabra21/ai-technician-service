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

_CONFIDENT   = 0.60
_POSSIBLE    = 0.40
_TOP2_MIN    = 0.30
_CLOSE_SURE  = 0.35
_CLOSE_POSS  = 0.20

_RULES = [
    # ── فرامل + رج/هز/رعشة ─────────────────────────────────
    (['فرامل','رج'],    'brakes'),  (['فرامل','هز'],    'brakes'),
    (['فرامل','رعش'],   'brakes'),  (['فرمل','رج'],     'brakes'),
    (['فرمل','هز'],     'brakes'),  (['احتكاك','فرامل'],'brakes'),
    (['صفار'],          'brakes'),  (['صفير'],          'brakes'),
    (['رجه','تقف'],     'brakes'),  (['برجه','تقف'],    'brakes'),
    (['رعشه','اشاره'],  'brakes'),  (['احتكاك','تقف'],  'brakes'),
    (['حكه','ببطء'],    'brakes'),  (['حكة','ببطء'],    'brakes'),
    # ── engine ─────────────────────────────────────────────
    (['طف','هدي'],      'engine'),  (['طف','ارفع'],     'engine'),
    (['بطل','هدي'],     'engine'),  (['بطل','ارفع'],    'engine'),
    (['سحب','بنزين'],   'engine'),  (['سحب','غاز'],     'engine'),
    (['صوت','مطلع'],    'engine'),  (['بيعلى','مطلع'],  'engine'),
    (['تعب','مطلع'],    'engine'),  (['تتعب','مطلع'],   'engine'),
    (['تاخر','سرعه'],   'engine'),  (['تاخر','بنزين'],  'engine'),
    (['فصل','لحظه'],    'engine'),  (['تقطع','ماشي'],   'engine'),
    (['صوت','غريب','قليله'], 'engine'),
    (['بتمشي','بتقل'],  'engine'),  (['بتتحرك','بتقل'], 'engine'),
    (['بتتحرك','ببطء'], 'engine'),
    (['تقيله','مسافه'], 'engine'),  (['صوتها','لكن','اعلي'], 'engine'),
    (['سلاسه','اقل'],   'engine'),  (['ناعمه','سواقه'], 'engine'),
    (['كركب'],          'engine'),
    (['اهتزاز','فجاه'], 'engine'),  (['اهتزاز','مفاجئ'], 'engine'),
    # ── exhaust ─────────────────────────────────────────────
    (['ريح','غريب'],    'exhaust'), (['ريح','بنزين'],   'exhaust'),
    (['ريح','شياط'],    'exhaust'), (['ريح','حريق'],    'exhaust'),
    (['ريح','محروق'],   'exhaust'), (['ريح','عادم'],    'exhaust'),
    # ── tires ───────────────────────────────────────────────
    (['طنين'],          'tires'),
    (['زنه','مستمره'],  'tires'),   (['زنه','سرعه'],    'tires'),
    (['صوت','زن'],      'tires'),   (['خبط','عجله'],    'tires'),
    # ── starting ────────────────────────────────────────────
    (['ساعات','بتدور'], 'starting'), (['ساعات','بتشتغل'], 'starting'),
    # ── alignment_balancing ──────────────────────────────────
    (['100','هز'], 'alignment_balancing'), (['90','هز'], 'alignment_balancing'),
    (['80','هز'],  'alignment_balancing'), (['80','رج'], 'alignment_balancing'),
    (['ثابته','طريق','سريع'],  'alignment_balancing'),
    (['ثابته','ملفات'],        'alignment_balancing'),
    (['رجرجه','هدي'],          'alignment_balancing'),
    (['رجرجه','اهدى'],         'alignment_balancing'),
    (['احتكاك','سرعه'],        'alignment_balancing'),
    # ── suspension ──────────────────────────────────────────
    (['طقطقه','الحركه'],       'suspension'),
    (['طقطقه','مع','الحركه'],  'suspension'),
    # ── cooling ─────────────────────────────────────────────
    (['تسخن','اداء'],  'cooling'), (['صوت','يسخن'],   'cooling'),
    (['صوت','تسخن'],   'cooling'), (['تسخن','تعب'],    'cooling'),
    'brakes',  (['فرامل','هز'],     'brakes'),
    (['فرامل','رعش'],    'brakes'),  (['فرمل','رج'],      'brakes'),
    (['فرمل','هز'],      'brakes'),  (['فرمل','رعش'],     'brakes'),
    (['فرمله','رج'],     'brakes'),  (['فرمله','هز'],     'brakes'),
    (['احتكاك','فرامل'], 'brakes'),  (['احتكاك','ببطء'],  'brakes'),

    # ── صفارة/صفير ────────────────────────────────────
    (['صفار'],           'brakes'),  (['صفير'],           'brakes'),

    # ── رجة + وقوف/فرملة فجأة ─────────────────────────
    (['رجه','تقف'],      'brakes'),  (['رجه','فجا'],      'brakes'),
    (['رجة','تقف'],      'brakes'),  (['رجه','وقفت'],     'brakes'),
    (['رجه','فرمل'],     'brakes'),

    # ── بتطفى/بتبطل + اهدي/ارفع ───────────────────────
    (['طف','هدي'],       'engine'),  (['طف','ارفع'],      'engine'),
    (['طف','اهدى'],      'engine'),  (['بطل','هدي'],      'engine'),
    (['بطل','ارفع'],     'engine'),

    # ── سحب ضعيف + بنزين/غاز ──────────────────────────
    (['سحب','بنزين'],    'engine'),  (['سحب','غاز'],      'engine'),

    # ── صوت/بيعلى + مطلع ──────────────────────────────
    (['صوت','مطلع'],     'engine'),  (['بيعلى','مطلع'],   'engine'),
    (['عيط','مطلع'],     'engine'),

    # ── تأخر في السرعة ────────────────────────────────
    (['تاخر','سرعه'],    'engine'),  (['تاخر','بنزين'],   'engine'),
    (['تاخير','بنزين'],  'engine'),

    # ── فصل/تقطع مؤقت ─────────────────────────────────
    (['فصل','لحظه'],     'engine'),  (['فصل','ترجع'],     'engine'),
    (['تقطع','ماشي'],    'engine'),  (['تقطع','ماشيه'],   'engine'),

    # ── صوت غريب + سرعة بطيئة ─────────────────────────
    (['صوت','غريب','قليله'],  'engine'),
    (['صوت','غريب','ببطء'],   'engine'),
    (['صوت','غريب','بطيء'],   'engine'),

    # ── بتمشي بتقل ────────────────────────────────────
    (['بتمشي','بتقل'],   'engine'),  (['ماشي','بتقل'],    'engine'),
    (['ماشيه','بتقل'],   'engine'),

    # ── تقيلة + مسافة ─────────────────────────────────
    (['تقيله','مسافه'],  'engine'),  (['تقيله','مسافة'],  'engine'),
    (['تقيلة','مسافه'],  'engine'),  (['تقيلة','مسافة'],  'engine'),

    # ── صوت بقى أعلى/مختلف + كويس/ماشية ──────────────
    (['صوت','بقى','اعلى'],   'engine'),
    (['صوت','مختلف','كويس'], 'engine'),
    (['صوت','اعلى','كويس'],  'engine'),

    # ── سلاسة أقل ─────────────────────────────────────
    (['سلاسه','اقل'],    'engine'),  (['سلاسة','أقل'],    'engine'),

    # ── مش ناعمة في السواقة ───────────────────────────
    (['ناعمه','سواقه'],  'engine'),  (['ناعمة','سواقة'],  'engine'),

    # ── ريحة → exhaust ────────────────────────────────
    (['ريح','غريب'],     'exhaust'), (['ريح','بنزين'],    'exhaust'),
    (['ريح','شياط'],     'exhaust'), (['ريح','حريق'],     'exhaust'),
    (['ريح','محروق'],    'exhaust'), (['ريح','عادم'],     'exhaust'),

    # ── زن/طنين → tires ───────────────────────────────
    (['طنين'],           'tires'),   (['صوت','زن'],       'tires'),

    # ── ساعات بتدور → starting ────────────────────────
    (['ساعات','بتدور'],  'starting'), (['ساعات','بتشتغل'], 'starting'),

    # ── أرقام سرعة + هز/رج → alignment ───────────────
    (['100','هز'], 'alignment_balancing'), (['100','رج'], 'alignment_balancing'),
    (['80','هز'],  'alignment_balancing'), (['80','رج'],  'alignment_balancing'),
    (['90','هز'],  'alignment_balancing'), (['120','هز'], 'alignment_balancing'),

    # ── مش ثابتة على طريق سريع → alignment ───────────
    (['ثابته','طريق','سريع'],  'alignment_balancing'),
    (['ثابت','طريق','سريع'],   'alignment_balancing'),
    (['مستقره','طريق','سريع'], 'alignment_balancing'),

    # ── تسخن + أداء/أبطأ/تعب → cooling ───────────────
    (['تسخن','اداء'],    'cooling'), (['تسخن','بطا'],     'cooling'),
    (['تسخن','تعب'],     'cooling'), (['تسخن','تضعف'],    'cooling'),

]

def _normalize(text: str) -> str:
    for a, b in [('ة','ه'),('ى','ي'),('أ','ا'),('إ','ا'),('آ','ا'),('ئ','ي'),('ؤ','و')]:
        text = text.replace(a, b)
    return text

def _preprocess(text: str) -> str:
    text = _normalize(text.lower().strip())
    text = re.sub(r'[،,؟?!.]', ' ', text)
    return re.sub(r'\s+', ' ', text)

def _apply_rules(clean: str):
    for kws, cat in _RULES:
        if all(k in clean for k in kws):
            return cat
    return None

def _make(svc_map, cat, conf):
    s = svc_map[cat]
    return RecommendedService(serviceId=s["id"], serviceName=s["name"], confidence=round(conf, 3))

def _analyze(text: str) -> AnalyzeProblemResponse:
    data     = _get_model()
    pipeline = data["pipeline"]
    svc_map  = data["services"]
    clean    = _preprocess(text)

    rule_cat = _apply_rules(clean)
    if rule_cat:
        return AnalyzeProblemResponse(
            status="success",
            recommendedServices=[_make(svc_map, rule_cat, 0.90)],
        )

    proba    = pipeline.predict_proba([clean])[0]
    classes  = pipeline.classes_
    idxs     = np.argsort(proba)[::-1]
    t1c, t1v = classes[idxs[0]], float(proba[idxs[0]])
    t2c, t2v = classes[idxs[1]], float(proba[idxs[1]])
    gap      = t1v - t2v

    if t1v < _POSSIBLE or t1c == "unknown":
        return AnalyzeProblemResponse(
            status="unknown",
            message="مش قدرنا نحدد المشكلة، ممكن توضح أكتر؟",
        )

    if t1v >= _CONFIDENT:
        svcs = [_make(svc_map, t1c, t1v)]
        if gap <= _CLOSE_SURE and t2v >= _TOP2_MIN and t2c != "unknown":
            svcs.append(_make(svc_map, t2c, t2v))
        return AnalyzeProblemResponse(status="success", recommendedServices=svcs)

    svcs = [_make(svc_map, t1c, t1v)]
    if gap <= _CLOSE_POSS and t2v >= _TOP2_MIN and t2c != "unknown":
        svcs.append(_make(svc_map, t2c, t2v))
    msg = "في احتمالين للمشكلة، التقني هيحدد بعد الفحص" if len(svcs) > 1 \
          else "مش متأكدين 100%، بس ده الأرجح"
    return AnalyzeProblemResponse(status="possible", recommendedServices=svcs, message=msg)


router = APIRouter()

@router.post("/api/analyze-problem", response_model=AnalyzeProblemResponse)
async def analyze_problem(request: AnalyzeProblemRequest):
    return _analyze(request.problemDescription)