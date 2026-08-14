# الدليل التقني الكامل للمستودع والنتائج

## 1. نطاق هذا المستودع

هذا المستودع هو الحزمة العملية للدراسة: كود الطرق، إعدادات التجارب، بيانات الدراسة المضمنة، النتائج المعتمدة، الأدلة التفصيلية المتاحة، أدوات إعادة التنفيذ، وفحوص السلامة. مواد التسليم للمجلة محفوظة خارج المستودع العملي في المجلد المستقل `01_Journal_Submission` ضمن حزمة التسليم الكاملة.

يجب التمييز بين مسارين لا يكتب أحدهما فوق الآخر:

- `results/main_results/`: سجل النتائج المعتمد والمستخدم في جداول الدراسة؛ محمي من الكتابة بواسطة أدوات التنفيذ.
- `results/execution_runs/`: الوجهة الوحيدة لأي تشغيل جديد. هذا المجلد مستبعد من Git، باستثناء ملف الشرح، حتى لا تختلط عمليات إعادة التنفيذ بالنتائج المعتمدة.

## 2. خريطة المستودع

| المسار | المحتوى |
|---|---|
| `src/asc_stream/` | Proposed، Fixed Rank، الطرق المقارنة، تحميل البيانات، التقييم، وضوابط مسارات الإخراج |
| `experiments/` | مشغلات التجارب والتجميع من ملفات التشغيل الجديدة |
| `configs/datasets/` | المصدر الوحيد لإعدادات كل داتا سيت في التجربة متعددة البيانات |
| `configs/controlled_representation_drift.json` | إعدادات تجربة الانحراف المضبوطة |
| `data/` | بيانات الدراسة المضمنة وManifest بصماتها ومسار تجهيز المصادر العامة |
| `results/main_results/` | النتائج المعتمدة ومصادر الجداول والتحليلات |
| `results/execution_runs/` | نتائج أي تنفيذ جديد؛ لا تُرفع تلقائياً مع Git |
| `docs/` | وصف سجل التنفيذ والبروتوكول الموحد والأدلة |
| `tests/` و`scripts/` | الاختبارات وفحوص البروتوكول والبيانات والنتائج والـManifest |
| `.github/workflows/ci.yml` | فحص آلي على Ubuntu وWindows باستخدام Python 3.11 |

## 3. البيانات المضمنة

يسجل `data/study_assets_manifest.json` المسار والحجم والشكل وSHA-256 لكل أصل دراسة. الأصول الرئيسية هي:

| الداتا سيت | الشكل المضمن |
|---|---:|
| CoverType | 581,012 × 54 |
| Electricity | 45,312 × 8 |
| TweetEval study representation | 59,899 × 256 |
| Synthetic GMM | 9,000 × 256 |

`python main.py verify-data` يتحقق من الأصول المضمنة مباشرة. أما `python main.py verify-canonical-data` فهو فحص إضافي لمسار المصدر العام بعد تنفيذ `setup-data`. تمثيل TweetEval العام القابل لإعادة التجهيز يستخدم 2048 سمة sparse، في حين أن أصل الدراسة المضمن هو تمثيل 256-D المثبت في Manifest؛ ولا يجوز الخلط بين المستويين.

## 4. خريطة النتائج المعتمدة

| جزء الدراسة | المصدر الرقمي |
|---|---|
| التجربة الأصلية متعددة البيانات: Proposed وFixed Rank وCluStream وDenStream وStreamKM++ | `results/main_results/multidataset/` |
| الإضافة اللاحقة: TWStream وFRA-ART على البيانات الأربع وتحت بروتوكول الدراسة نفسه | `results/main_results/modern_methods_extension/` |
| المقارنة المباشرة: Proposed الأصلي مع الطريقتين الحديثتين | `results/main_results/all_dataset_direct_comparison.csv` |
| تجربة الانحراف المضبوطة | `results/main_results/controlled/` |
| Ablation | `results/main_results/ablation/` |
| تحليل الرتبة على البيانات الأربع | `results/main_results/rank_all_datasets/` |
| تحليل الانحراف والاستعادة | `results/main_results/drift/` |
| الإحصاء والحساسية والموارد | `statistics/` و`sensitivity/` و`resources/` تحت `main_results` |

نتائج التجربة الأصلية تبقى كما هي. نتائج TWStream وFRA-ART إضافة لاحقة ولا تستبدل أياً من النتائج الأصلية. المقارنة المباشرة تستخدم Proposed من السجل الأصلي وتضيف صفوف الطريقتين الحديثتين. فحص `scripts/verify_main_results.py` يتأكد من ثبات قيم Proposed، واتساق خريطة الإضافة الحديثة، وإعادة تجميع سجلات controlled وablation، وعدد سجلات rank، ومقامات الاستعادة 10/10 و6/10.

## 5. إعادة التنفيذ الموحدة

المسار الرسمي لأي تنفيذ جديد هو:

```bash
python main.py run-primary
```

وهو ينفذ مصفوفة واحدة من 280 تشغيلًا:

- 4 datasets؛
- 7 methods: Proposed، Fixed Rank، CluStream، DenStream، StreamKM++، TWStream، FRA-ART؛
- 10 seeds: 7، 13، 19، 23، 31، 37، 41، 43، 47، 53.

كل طريقة تستخدم محمل البيانات وترتيب المشاهدات والـseed والتقييم المشترك نفسه. التحويلات التي تتطلبها طريقة بعينها تُطبّق بعد المعالجة المشتركة، وتُسجل قيمها المحلولة داخل ملف التشغيل الخام. لا يقرأ هذا المسار أرقامًا من `results/main_results/`.

بعد اكتمال المصفوفة المحددة، ينشئ التجميع من ملفات التنفيذ الجديدة فقط:

- `execution_manifest.json`؛
- `seedwise_results.csv`؛
- `dataset_method_summary.csv`؛
- `overall_equal_dataset_summary.csv`.

أوامر التحكم:

```bash
python main.py run-primary --dry-run
python main.py run-primary --resume
python main.py run-primary --dataset electricity --method proposed --seed 7
```

`--dry-run` يعرض الخطة ولا يشغل تجربة. `--resume` يتجاوز ملفات التشغيل الجديدة المكتملة. لا تُستبدل الملفات الموجودة افتراضياً، و`--force` لا يعمل إلا داخل `results/execution_runs/`. أما `--max-observations` فهو للاختبار الدخاني فقط وتُوسم مخرجاته `full_stream=false`، فلا تُعامل كنتيجة دراسة كاملة.

## 6. تحليل الرتبة

`python main.py run-rank` يبني Proposed نفسه المستخدم في `run-primary` ويستعمل المقيم نفسه، ثم يستخرج تفاصيل الرتبة من instance النموذج الرئيسي. لا توجد خوارزمية أو rank controller تشخيصية مستقلة. تذهب النتائج الجديدة إلى `results/execution_runs/rank_diagnostic/`.

## 7. التحقق قبل الاستخدام أو النشر

البيئة المدعومة هي Python 3.11. على Linux أو macOS:

```bash
python -m pip install -r requirements.txt
python main.py verify
```

وعلى Windows تُستخدم الأوامر نفسها من PowerShell أو الملف `SETUP_AND_VERIFY_WINDOWS.bat`.

يشمل `verify`:

1. سلامة بنية الحزمة واعتمادياتها؛
2. اتساق بروتوكول الدراسة؛
3. أشكال وبصمات بيانات الدراسة المضمنة؛
4. إعادة تجميع الأدلة الرقمية واتساق جداول النتائج؛
5. 179 بصمة للنتائج المحمية؛
6. Manifest ثنائي الاتجاه لكل ملفات الإصدار؛
7. ترجمة مصادر Python والاختبارات.

لا يقوم `verify` بتشغيل مصفوفة التجارب. تشغيل التجارب لا يحدث إلا عند طلب أحد أوامر `run-*` صراحة.

## 8. رفع المشروع إلى GitHub

ارفع محتويات مجلد `02_GitHub_Repository` نفسه ليكون جذر المستودع، لا ترفع المجلد الأب ولا ملفات المجلة. مثال:

```bash
git init
git add .
git status
git commit -m "Release complete adaptive stream clustering research package"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

قبل `git commit` يجب أن يظهر مجلد `results/main_results/rank_all_datasets/raw_logs/` ضمن الملفات المضافة؛ يوجد استثناء صريح له في `.gitignore`. يجب ألا تظهر cache files أو ملفات `results/execution_runs/` الجديدة.

أكبر أصل منفرد في الحزمة أصغر من حد GitHub البالغ 100 MiB للملف الواحد. عند استعمال واجهة الويب قد يكون رفع عدد كبير من الملفات غير عملي؛ يفضّل `git push` من سطر الأوامر.

لم تُضف رخصة استخدام تلقائياً؛ اختيار الرخصة قرار صاحب البحث. يمكن إبقاء المستودع خاصاً إلى حين اختيارها أو إضافة ملف `LICENSE` مناسب.

## 9. حدود التفسير الصحيح

- لا تُنشأ قيم خام تاريخية من المتوسطات والانحرافات المعيارية.
- لا تُقرب نتائج تنفيذ جديد بغرض جعلها مساوية لرقم منشور.
- لا تُكتب نتائج جديدة فوق السجل المعتمد.
- فهرس `docs/primary_execution_index.csv` يوثق 200 توليفة للتجربة الأصلية، أما المسار التنفيذي الموحد الحالي فينشئ 280 سجلًا خامًا عند تشغيله بالكامل.
- ملفات المجلة منفصلة عن الكود؛ الفصل مقصود حتى يبقى مستودع GitHub تقنياً ونظيفاً.

للتفاصيل الإضافية راجع `REPRODUCIBILITY.md` و`RESULTS.md` و`METHOD_SOURCES.md` و`docs/UNIFIED_REEXECUTION_PROTOCOL.md` و`docs/PRIMARY_EXECUTION_RECORD.md`.
