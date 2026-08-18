# مشروع Adaptive Stream Clustering — الحزمة العملية

هذه الحزمة تجمع كود الدراسة، إعدادات البروتوكول، بيانات الدراسة المضمنة،
النتائج المعتمدة، أدلة التدقيق، وأدوات إعادة التنفيذ والتحقق.

## التحقق

البيئة المدعومة هي Python 3.11:

```bash
python -m pip install -r requirements.txt
python main.py verify
```

يفحص `verify` بنية الحزمة، بروتوكول الدراسة، بصمات البيانات، اتساق النتائج،
بصمات النتائج المحمية، Manifest الإصدار، ترجمة ملفات Python، والاختبارات.

## تنظيم النتائج

- `results/main_results/`: أدلة الإصدار المحمية المستخدمة في المخطوطة.
- `results/execution_runs/`: مخرجات أي تشغيل جديد.
- `results/editor_round2_audit/`: أدلة التحقق الخاصة بملاحظات الجولة الثانية.

يشمل الإصدار تحديث TweetEval وSynthetic GMM الناتج من التحقق المتطابق، وهو
موثق في `docs/ROUND2_CORRECTION_20260817.md`.

## الإعدادات

المصدر الوحيد لإعدادات التجارب متعددة البيانات هو `configs/datasets/`.
أما التجربة المضبوطة فإعداداتها في
`configs/controlled_representation_drift.json`.

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
python main.py build-figures
```

`run-primary` هو المسار الرسمي لأي إعادة تنفيذ جديدة ويشغّل الطرق السبع ضمن
المصفوفة المعلنة، ويكتب النتائج فقط داخل `results/execution_runs/`.

## مقياس TweetEval/Synthetic GMM المتطابق

بالنسبة إلى القيم المتطابقة المستخدمة في Table 5 / Appendix B، فإن قيمة
ARI/NMI لكل seed هي المتوسط الحسابي عبر مقاطع التقييم اللاحقة لمرحلة التهيئة
بحجم 1000 مشاهدة. أما التقسيم النهائي للنافذة الكاملة فهو مقياس تدقيقي منفصل
ولا يمثل القيمة المبلّغ عنها في الجدولين.

## مستويا أدلة TweetEval

المسار العام الموصوف في المخطوطة يبني تمثيلاً نصياً sparse بعدد 2048 سمة
باستخدام HashingVectorizer ثم يطبق إسقاط النموذج. ولأغراض التنفيذ الذاتي تحتوي
الحزمة أيضاً تمثيل دراسة ثابتاً بحجم 59,899 × 256 ومثبتاً ببصمة SHA-256.

يُتحقق من أصل 256-D بوصفه أصل دراسة ثابتاً. أما إعادة البناء الصارمة من المصدر
العام فتستخدم `setup-data` والمسار canonical.

## التوثيق

راجع:
- `RESULTS.md`
- `REPRODUCIBILITY.md`
- `EVIDENCE_SCOPE.md`
- `docs/PRIMARY_EXECUTION_RECORD.md`
- `docs/ROUND2_CORRECTION_20260817.md`
- `docs/GITHUB_TECHNICAL_GUIDE_AR.md`

## نطاق دليل الرتبة

أثر التحقق المتطابق في الجولة الثانية مباشرةً على TweetEval وSynthetic GMM، ولذلك فإن صفوفهما الخاصة بالرتبة هي نتائج full-stream من التحقق المتطابق. أما CoverType وElectricity فتبقى قيمهما من rank diagnostic السابق لأن ملاحظة المحرر لم تكن عنهما. يوضح الملف `results/main_results/rank_all_datasets/rank_evidence_scope.csv` عدد المشاهدات ونطاق الدليل لكل صف، ولا تصف الحزمة الصفوف الأربع على أنها تجربة full-stream جديدة موحدة.

ملفات rank diagnostic لا تحتوي عمداً على ARI/NMI الخاصة بالمخطوطة. مصدر جودة Table 5/Appendix B هو `results/main_results/multidataset/` وملفات التحقق المتطابق داخل `results/editor_round2_audit/`.

حد الدليل بين TweetEval ذي 2048 سمة والمسار المضمن 256-D موثق صراحة في `data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md` و`.json` من دون ادعاء سلسلة اشتقاق غير مثبتة.
