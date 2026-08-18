# الدليل التقني للمستودع والنتائج

## 1. نطاق المستودع

المستودع هو الحزمة العملية للدراسة: الكود، الإعدادات، بيانات الدراسة، النتائج
المحمية، أدلة التدقيق، وأدوات إعادة التنفيذ والتحقق. ملفات تسليم المجلة تبقى
منفصلة عن جذر المستودع.

المساران الرئيسيان:
- `results/main_results/`: النتائج المحمية المستخدمة في الإصدار.
- `results/execution_runs/`: مخرجات أي تشغيل جديد، ولا يجوز أن تكتب فوق
  `main_results`.

## 2. خريطة أساسية

| المسار | المحتوى |
|---|---|
| `src/asc_stream/` | Proposed وFixed Rank والطرق المقارنة |
| `experiments/` | مشغلات التجارب |
| `configs/datasets/` | إعدادات الداتا سيت الأساسية |
| `configs/controlled_representation_drift.json` | التجربة المضبوطة |
| `data/` | بيانات الدراسة وManifest البصمات |
| `results/main_results/` | مصادر جداول الإصدار |
| `results/editor_round2_audit/` | أدلة التحقق التحريري |
| `results/execution_runs/` | التشغيلات الجديدة |
| `tests/` و`scripts/` | الاختبارات وفحوص السلامة |

## 3. البيانات

الأصول المضمنة:
- CoverType: 581,012 × 54
- Electricity: 45,312 × 8
- TweetEval packaged study representation: 59,899 × 256
- Synthetic GMM: 9,000 × 256

توجد بصمات SHA-256 في `data/study_assets_manifest.json`.

### TweetEval

يجب التمييز بين مستويين:
1. المسار العام canonical: train → validation → test، وHashingVectorizer
   sparse بعدد 2048 سمة، ثم إسقاط النموذج.
2. المسار الذاتي study: أصل دراسة ثابت 59,899 × 256، مثبت بالبصمة، ويعامل
   كتمثيل projected محفوظ مسبقاً عند واجهة الإسقاط.

لا تدعي الحزمة أن ملف 256-D يمكن إعادة اشتقاقه تشفيرياً من 2048-D بمجرد وجوده.
للتحقق الصارم من المصدر العام استخدم:
`python main.py setup-data`
ثم المسار canonical.

## 4. خريطة النتائج

- `results/main_results/multidataset/`: جداول النتائج الأساسية.
- `results/main_results/modern_methods_extension/`: TWStream وFRA-ART للسياق
  الوصفي على البيانات الأربع.
- `results/main_results/controlled/`: المقارنة المضبوطة ذات 40 تشغيلًا.
- `results/main_results/ablation/`: 60 تشغيلًا.
- `results/main_results/rank_all_datasets/`: تحليل الرتبة عبر أربع بيانات ×
  عشر بذور.
- `results/editor_round2_audit/`: التحقق المتطابق من TweetEval/Synthetic GMM
  وفحوص الزمن والمقارنة الحديثة.

القيم المعتمدة بعد التحقق:
- TweetEval: Proposed = Fixed Rank في ARI/NMI والرتبة تبقى 8.
- Synthetic GMM Proposed: ARI 0.765705 ± 0.068506، NMI 0.779494 ± 0.060014،
  rank 11.6468 ± 0.3060، range 8–18.
- Synthetic GMM Fixed Rank: ARI 0.722984 ± 0.059987،
  NMI 0.736730 ± 0.056811، والرتبة 8.

## 5. تعريف مقياس الجودة المتطابق

في TweetEval وSynthetic GMM المتطابقين، القيم المبلّغ عنها في Table 5 /
Appendix B ليست جودة تقسيم نهائي واحد. لكل seed:
- أول 1000 مشاهدة هي initialization.
- يحسب ARI/NMI على مقاطع التقييم اللاحقة بحجم 1000 مشاهدة.
- تؤخذ القيمة الحسابية المتوسطة لهذه المقاطع.
- بعد ذلك تلخص البذور العشر بالمتوسط ± SD.

أما `complete_window_final_*` فهو تشخيص مستقل محفوظ للتدقيق، ولا يستبدل مقياس
المخطوطة.

## 6. إعادة التنفيذ الموحدة

```bash
python main.py run-primary
```

ينفذ 4 datasets × 7 methods × 10 seeds = 280 تشغيلًا. يكتب فقط داخل
`results/execution_runs/`.

أوامر مفيدة:
```bash
python main.py run-primary --dry-run
python main.py run-primary --resume
python main.py run-rank
python main.py run-controlled
```

## 7. التحقق

```bash
python main.py verify
```

يفحص البروتوكول والبيانات والنتائج والبصمات والـManifest والاختبارات.

بعد أي تحديث مقصود لأدلة الإصدار:
```bash
python scripts/verify_main_results.py
python scripts/verify_protected_results.py --update-reference
python scripts/generate_manifest.py
python main.py verify
```

يجب تنفيذ الفحص الدلالي للنتائج **قبل** اعتماد بصمات النتائج الجديدة.

## 8. الجولة الثانية

يوثق `docs/ROUND2_CORRECTION_20260817.md` التحقق المتطابق والتغييرات التي
أصبحت جزءاً من إصدار النتائج. لا يقدم الملف تخميناً عن سبب برمجي تاريخي؛
المهم هو أن القيم النهائية المستخدمة في المخطوطة مرتبطة مباشرة بالتنفيذ
المتطابق الموثق.

## 9. قبل الرفع إلى GitHub

- تأكد من عدم وجود `__pycache__` أو ملفات backup.
- لا ترفع تشغيلات جديدة من `results/execution_runs/`.
- راجع `git diff` قبل commit.
- شغّل `python main.py verify`.

## 10. نطاق Rank Diagnostic

لا يجوز تفسير الصفوف الأربع في Table 10 على أنها إعادة full-stream موحدة أُجريت كلها في الجولة الثانية. TweetEval وSynthetic GMM هما الصفان اللذان أُعيدا التحقق منهما full-stream بسبب ملاحظة المحرر. CoverType وElectricity يحتفظان بrank diagnostic السابق. يحدد `results/main_results/rank_all_datasets/rank_evidence_scope.csv` التغطية العددية ونوع الدليل لكل صف.

كما أزيلت ARI/NMI من ملفات rank release-facing حتى لا تختلط مع قيم Table 5/Appendix B. ويشرح `data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md` حد الدليل بين 2048-D canonical و256-D packaged.
