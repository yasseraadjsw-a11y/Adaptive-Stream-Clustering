# مشروع Adaptive Stream Clustering — الحزمة العملية النهائية

هذه الحزمة تجمع الكود، بيانات الدراسة المضمنة، النتائج الأساسية، ملفات النتائج التفصيلية، أدوات التنفيذ، والتحقق الآلي.

الدليل المركزي المفصل للمستودع والنتائج وإعادة التنفيذ والرفع إلى GitHub موجود في `docs/GITHUB_TECHNICAL_GUIDE_AR.md`. مواد التسليم للمجلة منفصلة عن هذا المستودع العملي.

## التحقق الأول

البيئة المعتمدة هي Python 3.11:

```bash
python -m pip install -r requirements.txt
python main.py verify
```

أمر `verify` يترجم ملفات Python، ويفحص بنية الحزمة، وبروتوكول الدراسة الموحد، وبصمات البيانات، وتطابق البحث مع النتائج، وبصمات النتائج المحمية، وManifest ثنائي الاتجاه، ثم يشغل جميع الاختبارات.

## النتائج

نتائج البحث المستخدمة في المخطوطة محفوظة داخل `results/main_results/` ومحميّة من الكتابة. كل تنفيذ جديد يذهب إلى `results/execution_runs/`.

يوجد داخل `docs/primary_execution_index.csv` فهرس التجربة التاريخية ذات 200 توليفة: 4 بيانات × 5 طرق × 10 بذور. أما مسار إعادة التنفيذ الموحد الجديد فيغطي 280 توليفة: 4 بيانات × 7 طرق × 10 بذور.

## الإعدادات

المصدر الوحيد لإعدادات الـPrimary هو `configs/datasets/`. لذلك نفس القيم المكتوبة في Table 4 هي التي يقرأها `run-primary` فعلياً. بروتوكول التجربة المضبوطة موجود في `configs/controlled_representation_drift.json`.

## أهم الأوامر

```bash
python main.py verify
python main.py verify-data
python main.py verify-canonical-data
python main.py rebuild-controlled-data
python main.py run-controlled
python main.py run-primary
python main.py run-rank
python main.py run-ablation
python main.py run-drift
python main.py run-sensitivity
python main.py profile-resources
```

`run-primary` هو المسار الرسمي لأي إعادة تنفيذ جديدة؛ فهو يشغّل الطرق السبع على البيانات الأربع والبذور العشر، ثم ينشئ Manifest للاكتمال، وجدول النتائج الخام لكل بذرة، وملخص كل طريقة/داتا سيت، والملخص الكلي من ملفات التنفيذ الجديدة فقط. لا يقرأ هذا المسار أي رقم من `main_results`. بقي `run-modern-all` كأداة تشخيص للطرق الحديثة وحدها ولا يدمج Proposed محفوظًا. أما `run-rank` فيبني نفس Proposed الرئيسي ويستخدم نفس الـevaluator ويكتب الأدلة الجديدة داخل `results/execution_runs/rank_diagnostic/`، ولا يوجد controller منفصل للرتبة.

الأمر `python main.py run-primary --dry-run` يعرض خطة الـ280 تشغيلًا من دون تنفيذ، و`--resume` يستكمل التنفيذ المنقطع.

## البيانات

كل أصل دراسة مضمن له SHA-256 وحجم وشكل متوقع. الحزمة تحتوي CoverType وElectricity كاملتين، وتدفق Synthetic GMM الفعلي 9000×256 المستخدم في التجربة الأصلية وفي التحليل المضبوط اللاحق، وتمثيل TweetEval المستخدم في الحزمة. كما يوجد مسار إعادة تجهيز TweetEval من المصدر الرسمي إلى 2048-D وفق الإعدادات المعلنة في البحث.

`verify-data` يفحص بيانات الحزمة مباشرة، و`verify-canonical-data` هو فحص provenance إضافي؛ وبالنسبة إلى TweetEval العام يتطلب تجهيز المصدر أولاً.
