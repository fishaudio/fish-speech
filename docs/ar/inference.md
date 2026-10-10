# الاستنتاج

يتطلب نموذج Fish Audio S2 ذاكرة فيديو (VRAM) كبيرة. نوصي باستخدام وحدة معالجة رسومات (GPU) بسعة 24 جيجابايت على الأقل للاستنتاج.

## تحميل الأوزان

أولاً ، تحتاج إلى تحميل أوزان النموذج:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## الاستنتاج عبر خط الأوامر

!!! note
    إذا كنت تخطط لترك النموذج يختار نغمة الصوت عشوائيًا ، فيمكنك تخطي هذه الخطوة.

### 1. الحصول على رموز VQ من الصوت المرجعي

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

يجب أن تحصل على `fake.npy` و `fake.wav`.

### 2. توليد الرموز الدلالية (Semantic tokens) من النص:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "النص الذي تريد تحويله" \
    --prompt-text "النص المرجعي الخاص بك" \
    --prompt-tokens "fake.npy" \
    # --compile
```

سيقوم هذا الأمر بإنشاء ملف `codes_N` في دليل العمل ، حيث N هو عدد صحيح يبدأ من 0.

!!! note
    قد ترغب في استخدام `--compile` لدمج نوى CUDA لاستنتاج أسرع. ومع ذلك ، نوصي باستخدام تحسين تسريع الاستنتاج sglang الخاص بنا.
    بالمقابل ، إذا كنت لا تخطط لاستخدام التسريع ، يمكنك التعليق على معلمة `--compile`.

!!! info
    بالنسبة لوحدات معالجة الرسومات التي لا تدعم bf16 ، قد تحتاج إلى استخدام معلمة `--half`.

### 3. توليد الصوت من الرموز الدلالية:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

بعد ذلك ستحصل على ملف `fake.wav`.

## S3-drama

يستخدم S3-drama سطر الأوامر وخادم API نفسيهما المستخدمين مع S2-Pro، ويُحدَّد نوع النموذج تلقائيًا من ملف `config.json` في مجلد الأوزان.

!!! warning
    يعمل S3-drama على وحدات معالجة الرسومات من NVIDIA فقط. لا يدعم وحدة المعالجة المركزية (CPU) ولا Apple MPS ولا AMD ROCm، كما لا يدعم `--compile`.

| GPU | طريقة التشغيل | ذاكرة GPU |
| --- | --- | --- |
| SM89 أو أحدث (L40، H100، H20) | تبقى خبراء MoE بصيغة FP8، مع فك ترميز باستخدام CUDA graphs، على GPU واحدة | نحو 41 GB |
| وحدات أقدم (V100، A100) | يُلغى تكميم خبراء MoE وتُوزَّع على عدة وحدات GPU | نحو 70 GB إجمالًا |

### تحميل الأوزان

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

إذا لم يحتوِ مجلد الأوزان على `codec.pth`، فسيُستخدم مُرمِّز S2-Pro الموجود في `checkpoints/s2-pro/codec.pth`.

### الاستنتاج عبر سطر الأوامر

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "النص الذي تريد تحويله" \
    --prompt-audio "reference.wav" \
    --prompt-text "النص المرجعي الخاص بك" \
    --output "output.wav" \
    --device cuda:0
```

يكتب الخيار `--output` الصوت بعد فك ترميزه مباشرةً. على وحدات GPU الأقدم، حدِّد عدة أجهزة، مثل `--device cuda:0,cuda:1,cuda:2,cuda:3`؛ ويتولى الجهاز الأخير مُفكِّك ترميز الصوت والمُرمِّز. أما `--device cuda` وحده فيستخدم حتى أول أربع وحدات GPU مرئية.

### خادم API

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

مع Docker، شغِّل صورة الخادم مع أوزان S3. على البطاقات SM89 أو الأحدث، اعرض وحدة GPU واحدة فقط كما في المثال أدناه؛ وعلى البطاقات الأقدم استخدم `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s3-drama-fp8/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### الاختلافات عن S2-Pro

- يُولَّد النص كاملًا دفعة واحدة. افتراضيًا، ينتج كل طلب 2048 إطارًا كحد أقصى (نحو 95 ثانية من الصوت)، وتُقيَّد قيمة `max_new_tokens` الأكبر بـ 10,240 إطارًا مطروحًا منها طول الموجّه (prompt). قسِّم النصوص الأطول على عدة طلبات.
- لا تأثير للخيارين `chunk_length` و`repetition_penalty`؛ إذ يتضمن S3 أخذ عينات مدركًا للتكرار. إعدادات أخذ العينات الافتراضية هي temperature 1.0 وtop-p 0.9 وtop-k 30.
- عند تفعيل `streaming`، يصل الصوت دفعة واحدة بعد انتهاء التوليد.
- يفرض `FISH_S3_FP8_EXPERTS=0` أو `1` استخدام خبراء غير مكمَّمين أو بصيغة FP8، ويُعطِّل `FISH_S3_CUDA_GRAPH=0` فك الترميز باستخدام CUDA graphs.

## استنتاج WebUI

### Awesome WebUI

تعد Awesome WebUI واجهة ويب حديثة تعتمد على TypeScript، وتوفر ميزات أغنى وتجربة مستخدم أفضل.

**بناء WebUI:**

يجب أن يكون لديك Node.js و npm مثبتين على جهازك المحلي أو الخادم.

1. ادخل إلى دليل `awesome_webui`:
   ```bash
   cd awesome_webui
   ```
2. تثبيت التبعيات:
   ```bash
   npm install
   ```
3. بناء WebUI:
   ```bash
   npm run build
   ```

**بدء تشغيل خادم الخلفية:**

بعد بناء WebUI، عد إلى دليل جذر المشروع وقم بتشغيل خادم API:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**الوصول:**

بمجرد تشغيل الخادم، يمكنك الوصول إليه عبر المتصفح على العنوان التالي:
`http://localhost:8888/ui`
