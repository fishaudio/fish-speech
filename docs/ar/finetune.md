# الضبط الدقيق (Fine-tuning)

من الواضح أنك عندما فتحت هذه الصفحة، لم تكن راضيًا عن أداء النموذج المدرب مسبقًا في وضع zero-shot. أنت ترغب في إجراء ضبط دقيق لنموذج لتحسين أدائه على مجموعة البيانات الخاصة بك.

في الإصدار الحالي، ما عليك سوى إجراء الضبط الدقيق لجزء 'LLAMA'.

## الضبط الدقيق لـ LLAMA
### 1. إعداد مجموعة البيانات

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```

تحتاج إلى تحويل مجموعة البيانات الخاصة بك إلى التنسيق أعلاه ووضعها تحت مجلد `data`. يمكن أن يكون للملف الصوتي الامتدادات `.mp3`، `.wav`، أو `.flac`، ويجب أن يكون لملف التعليقات التوضيحية الامتداد `.lab`.

!!! info "تنسيق مجموعة البيانات"
    يحتاج ملف التعليقات التوضيحية `.lab` فقط إلى احتواء النص المكتوب للمقطع الصوتي، دون الحاجة إلى تنسيق خاص. على سبيل المثال، إذا كان محتوى `hi.mp3` هو "مرحبًا، وداعًا"، فسيحتوي ملف `hi.lab` على سطر واحد من النص: "مرحبًا، وداعًا".

!!! warning "تحذير"
    يوصى بتطبيق تسوية جهارة الصوت (loudness normalization) على مجموعة البيانات. يمكنك استخدام [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) للقيام بذلك.
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. الاستخراج الدفعي للرموز الدلالية (semantic tokens)

تأكد من أنك قمت بتنزيل أوزان VQGAN. إذا لم تكن قد فعلت، قم بتشغيل الأمر التالي:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

يمكنك بعد ذلك تشغيل الأمر التالي لاستخراج الرموز الدلالية:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

!!! note "ملاحظة"
    يمكنك ضبط `--num-workers` و `--batch-size` لزيادة سرعة الاستخراج، ولكن يرجى التأكد من عدم تجاوز حد ذاكرة وحدة معالجة الرسومات (GPU) الخاصة بك.

سيقوم هذا الأمر بإنشاء ملفات `.npy` في مجلد `data`، كما هو موضح أدناه:

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 21.15-26.44.npy
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 27.51-29.98.npy
│   ├── 30.1-32.71.lab
│   ├── 30.1-32.71.mp3
│   └── 30.1-32.71.npy
└── SPK2
    ├── 38.79-40.85.lab
    ├── 38.79-40.85.mp3
    └── 38.79-40.85.npy
```

### 3. حزم مجموعة البيانات في protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

بعد انتهاء تنفيذ الأمر، يجب أن ترى ملف `protos` في مجلد `data`.

### 4. أخيرًا، الضبط الدقيق باستخدام LoRA

بالمثل، تأكد من أنك قمت بتنزيل أوزان `LLAMA`. إذا لم تكن قد فعلت، قم بتشغيل الأمر التالي:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

أخيرًا، يمكنك بدء الضبط الدقيق عن طريق تشغيل الأمر التالي:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note "ملاحظة"
    يمكنك تعديل معلمات التدريب مثل `batch_size`، `gradient_accumulation_steps`، وما إلى ذلك لتناسب ذاكرة وحدة معالجة الرسومات الخاصة بك عن طريق تعديل `fish_speech/configs/text2semantic_finetune.yaml`.

!!! note "ملاحظة"
    لمستخدمي Windows، يمكنك استخدام `trainer.strategy.process_group_backend=gloo` لتجنب مشكلات `nccl`.

بعد اكتمال التدريب، يمكنك الرجوع إلى قسم [الاستدلال (inference)](inference.md) لاختبار نموذجك.

!!! info "معلومات"
    بشكل افتراضي، سيتعلم النموذج فقط أنماط كلام المتحدث وليس جرس الصوت (timbre). لا تزال بحاجة إلى استخدام التلقينات (prompts) لضمان استقرار جرس الصوت.
    إذا كنت ترغب في تعلم جرس الصوت، يمكنك زيادة عدد خطوات التدريب، ولكن هذا قد يؤدي إلى الإفراط في التخصيص (overfitting).

بعد التدريب، تحتاج إلى تحويل أوزان LoRA إلى أوزان عادية قبل إجراء الاستدلال.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/openaudio-s1-mini \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/openaudio-s1-mini-yth-lora/
```
!!! note "ملاحظة"
    يمكنك أيضًا تجربة نقاط تحقق (checkpoints) أخرى. نقترح استخدام أقدم نقطة تحقق تلبي متطلباتك، حيث إنها غالبًا ما تؤدي أداءً أفضل على البيانات خارج التوزيع (OOD).
