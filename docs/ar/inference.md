# الاستنتاج

نظرًا لأن نموذج vocoder قد تغير، تحتاج إلى VRAM أكثر من ذي قبل، يُنصح بـ 12GB للاستنتاج السلس.

ندعم سطر الأوامر و HTTP API و WebUI للاستنتاج، يمكنك اختيار أي طريقة تفضلها.

## تحميل الأوزان

أولاً تحتاج إلى تحميل أوزان النموذج:

```bash
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## استنتاج سطر الأوامر

!!! note
    إذا كنت تخطط لترك النموذج يختار نبرة صوت عشوائياً، يمكنك تخطي هذه الخطوة.

### 1. الحصول على رموز VQ من الصوت المرجعي

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

يجب أن تحصل على `fake.npy` و `fake.wav`.

### 2. إنتاج الرموز الدلالية من النص:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "النص الذي تريد تحويله" \
    --prompt-text "النص المرجعي الخاص بك" \
    --prompt-tokens "fake.npy" \
    --compile
```

هذا الأمر سينشئ ملف `codes_N` في دليل العمل، حيث N هو عدد صحيح يبدأ من 0.

!!! note
    قد ترغب في استخدام `--compile` لدمج نوى CUDA للاستنتاج الأسرع (~15 رمز/ثانية -> ~150 رمز/ثانية، على GPU RTX 4090).
    وفقاً لذلك، إذا كنت لا تخطط لاستخدام التسريع، يمكنك التعليق على معامل `--compile`.

!!! info
    بالنسبة لوحدات GPU التي لا تدعم bf16، قد تحتاج إلى استخدام معامل `--half`.

### 3. إنتاج الأصوات من الرموز الدلالية:

!!! warning "تحذير مستقبلي"
    لقد احتفظنا بإمكانية الوصول إلى الواجهة من المسار الأصلي (tools/vqgan/inference.py)، لكن هذه الواجهة قد تُزال في الإصدارات اللاحقة، لذا يرجى تغيير الكود الخاص بك في أقرب وقت ممكن.

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy"
```

## استنتاج HTTP API

نوفر HTTP API للاستنتاج. يمكنك استخدام الأمر التالي لبدء الخادم:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> إذا كنت تريد تسريع الاستنتاج، يمكنك إضافة معامل `--compile`.

بعد ذلك، يمكنك عرض واختبار API على http://127.0.0.1:8080/.

## استنتاج GUI 
[تحميل العميل](https://github.com/AnyaCoder/fish-speech-gui/releases)

## استنتاج WebUI

يمكنك بدء WebUI باستخدام الأمر التالي:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

أو ببساطة

```bash
python -m tools.run_webui
```
> إذا كنت تريد تسريع الاستنتاج، يمكنك إضافة معامل `--compile`.

!!! note
    يمكنك حفظ ملف التسمية وملف الصوت المرجعي مسبقاً في مجلد `references` في الدليل الرئيسي (الذي تحتاج إلى إنشاؤه بنفسك)، بحيث يمكنك استدعاؤها مباشرة في WebUI.

!!! note
    يمكنك استخدام متغيرات بيئة Gradio، مثل `GRADIO_SHARE`، `GRADIO_SERVER_PORT`، `GRADIO_SERVER_NAME` لتكوين WebUI.

استمتع!
