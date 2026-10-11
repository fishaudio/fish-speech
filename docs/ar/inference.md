# الاستنتاج

يوفر Fish Audio نموذجين: **S2-Pro** و **Drama 3**. يتشاركان في نفس سطر الأوامر وخادم API، ويُستنتج نوع النموذج من ملف `config.json` في مجلد الأوزان، لذا تم تنظيم كل قسم أدناه على شكل "المهمة → أمر كل نموذج".

أنهِ [التثبيت](install.md) أولاً، ثم حمّل [أوزان النموذج](install.md#تحميل-الأوزان).

!!! warning "قيود تشغيل Drama 3"
    يعمل Drama 3 على بطاقات NVIDIA فقط: لا يدعم CPU أو Apple MPS أو AMD ROCm، ولا يدعم `--compile`. راجع [المتطلبات](install.md#المتطلبات) لمعرفة ذاكرة GPU.

!!! note
    بطاقات مثل V100 والأقدم من SM80 لا تدعم bfloat16، فيتحول Drama 3 إلى float16 (مع إبقاء التيار المتبقي fp32). لم يُختبر هذا المسار على أجهزة فعلية بعد؛ إذا ظهر لديك صمت أو ضوضاء أو خطأ "non-finite logits"، يرجى فتح issue.

## الاستنتاج عبر خط الأوامر

### S2-Pro

يستخدم S2-Pro خطوة من ثلاث مراحل: "رموز VQ → رموز دلالية → صوت".

!!! note
    إذا كنت تريد أن يختار النموذج لون الصوت عشوائيًا، يمكنك تخطي الخطوة 1.

**1. الحصول على رموز VQ من الصوت المرجعي**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

يجب أن تحصل على ملفي `fake.npy` و `fake.wav`.

**2. توليد الرموز الدلالية (Semantic tokens) من النص:**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "النص الذي تريد تحويله" \
    --prompt-text "النص المرجعي" \
    --prompt-tokens "fake.npy" \
    # --compile
```

سينشئ هذا الأمر ملف `codes_N` في مجلد العمل، حيث N عدد صحيح يبدأ من 0.

!!! note
    قد ترغب في استخدام `--compile` لدمج نوى CUDA من أجل استنتاج أسرع، لكننا نوصي باستخدام تحسين الاستنتاج sglang.
    إذا كنت لا تخطط لاستخدام التسريع، يمكنك وضع `--compile` كتعليق.

!!! info
    على البطاقات التي لا تدعم bf16، قد تحتاج إلى استخدام `--half`.

**3. توليد الصوت من الرموز الدلالية:**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

بعد ذلك ستحصل على ملف `fake.wav`.

### Drama 3

ينجز Drama 3 الأمر بأمر واحد، ويكتب `--output` الصوت المفكوك ترميزه مباشرة:

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "النص الذي تريد تحويله" \
    --prompt-audio "reference.wav" \
    --prompt-text "النص المرجعي" \
    --output "output.wav" \
    --device cuda:0
```

على البطاقات الأقدم، يمكنك تحديد عدة بطاقات، مثل `--device cuda:0,cuda:1,cuda:2,cuda:3`؛ وتتولى البطاقة الأخيرة مفكك ترميز الصوت والـ codec. عند كتابة `--device cuda` فقط، يستخدم ما يصل إلى أول أربع بطاقات مرئية.

## خادم API

### تشغيل S2-Pro

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### تشغيل Drama 3

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    لا يدعم Drama 3 خيار `--compile`. على البطاقات الأقدم، اذكر عدة بطاقات عبر `--device`؛ وتتولى البطاقة الأخيرة مفكك ترميز الصوت والـ codec.

خيارات تشغيل مشتركة بين النموذجين:

- `--compile`: تمكين تحسين `torch.compile` (لـ S2-Pro فقط)
- `--half`: استخدام وضع fp16
- `--api-key`: طلب التوثيق بـ bearer token
- `--workers`: تعيين عدد عمليات العمل
- `--max-text-length`: تحديد طول نص الطلب الواحد؛ ويشير 0 إلى عدم وجود حد

فحص الصحة:

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

تشير الاستجابة `{"status":"ok"}` إلى أن الخدمة تعمل بشكل طبيعي.

### نقاط نهاية API

| الطريقة | المسار | الوصف |
| --- | --- | --- |
| `POST` | `/v1/tts` | تحويل النص إلى كلام؛ ويكون جسم الطلب بتنسيق msgpack |
| `GET` | `/v1/tts/defaults` | قيم أخذ العينات الافتراضية للنموذج المحمّل |
| `GET` | `/v1/health` | فحص الصحة |
| `POST` | `/v1/vqgan/encode` | الصوت المرجعي ← رموز VQ |
| `POST` | `/v1/vqgan/decode` | رموز VQ ← صوت |
| `POST` | `/v1/references/add` | حفظ صوت مرجعي؛ حقول النموذج `id` و `audio` (ملف) و `text` |
| `GET` | `/v1/references/list` | عرض الأصوات المحفوظة |
| `POST` | `/v1/references/update` | إعادة تسمية؛ حقلا `old_reference_id` و `new_reference_id` |
| `DELETE` | `/v1/references/delete` | الحذف؛ حقل `reference_id` |
| `GET` | `/ui` | واجهة WebUI (مدمجة في صورة Docker؛ يجب بناؤها عند التشغيل من المصدر) |

بعد حفظ الصوت، استخدمه عبر `--reference_id` (يحتاج حقل الصوت إلى نوع MIME، وإلا لن يتعرف عليه الخادم):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="النص المكتوب للصوت المرجعي"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### إرسال طلب

يعتمد النموذج المستخدم على الأوزان التي حمّلها الخادم عند بدء التشغيل، ولا يوجد حقل للنموذج داخل الطلب.

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

المعاملات الشائعة:

| المعامل | الوصف |
| --- | --- |
| `--reference_audio` / `--reference_text` | الصوت المرجعي ونصّه المكتوب، ويجب أن يكون عدد القيم متساويًا |
| `--reference_speaker` | معرّف المتحدث (0–15) لكل ملف صوتي مرجعي، ويُكتب كبادئة `<\|speaker:N\|>` (الاستخدام الرئيسي في Drama 3) |
| `--reference_id` | استخدام صوت مرجعي محفوظ، مثل `my-speaker` |
| `--format` | تنسيق الإخراج: `wav` أو `pcm` أو `mp3` أو `opus`، والافتراضي `wav` |
| `--streaming` | إرجاع الصوت بشكل متدفق |
| `--single_chunk` | توليد النص كاملاً دفعة واحدة دون تقسيم (لـ Drama 3 فقط) |
| `--chunk_length` | طول المقطع، 200 افتراضيًا (لـ S2-Pro فقط) |
| `--max_new_tokens` | الحد الأقصى للإطارات في كل مقطع؛ وإن لم يُحدد يُستخدم افتراض الخادم 1024، والقيمة 0 تعني حد النموذج نفسه |
| `--temperature` / `--top_p` | أخذ العينات؛ وإن لم يُحددا تُستخدم افتراضات النموذج (0.7 / 0.7) |
| `--seed` | تثبيت البذرة العشوائية (لا يمكن استخدامها لتثبيت لون الصوت) |
| `--api_key` | مطلوب عند تشغيل الخادم مع `--api-key` |
| `--no-play` | عدم تشغيل الصوت بعد التوليد (يتم التشغيل افتراضيًا؛ مطلوب على الأجهزة دون بطاقة صوت) |

!!! note
    تم الإبقاء على `repetition_penalty` من أجل التوافق فقط، فكل من S2-Pro و Drama 3 يتجاهلانه ويستخدمان أخذ عينات مدرك للتكرار مدمج.

!!! tip
    لتشغيل الخادم باستخدام Docker، راجع [إعداد Docker](install.md#إعداد-docker) (صورة الخادم تتضمن واجهة WebUI بالفعل: افتح `http://localhost:8080/ui` بعد تشغيلها).

## الاختلافات بين النموذجين

| البند | S2-Pro | Drama 3 |
| --- | --- | --- |
| العتاد | GPU / CPU | NVIDIA GPU فقط، دون `--compile` |
| النص الطويل | يُقسَّم حسب `chunk_length` | يُقسَّم تلقائيًا إلى مقاطع من نحو 300 بايت (UTF-8) بحد أقصى ثلاثة متحدثين، مع القطع عند نهايات الجمل قدر الإمكان؛ ويُعاد إدخال المقطع الأول والأحدث كسياق |
| حد الطول في المرة الواحدة | طول السياق | 1024 إطارًا لكل مقطع (نحو 47 ثانية)؛ و2048 إطارًا (نحو 95 ثانية) مع `single_chunk`؛ ويحدهما جميعًا 10,240 إطارًا ناقص طول الموجه |
| المتحدثون | ترتيب الملفات المرجعية يحدد الدور | علامات `<\|speaker:0\|>`–`<\|speaker:15\|>` تحدد الدور، ولا أهمية لترتيب المراجع |
| `chunk_length` | يعمل | لا يعمل |
| افتراضات أخذ العينات | temperature 0.7، top-p 0.7، top-k 30 | نفسها |
| `repetition_penalty` | مُتجاهَل | مُتجاهَل |

يحتوي Drama 3 أيضًا على متغيري بيئة: `FISH_DRAMA3_FP8_EXPERTS=0` أو `1` يفرض الخبراء غير المكمَّمين أو بتنسيق FP8، و`FISH_DRAMA3_CUDA_GRAPH=0` يعطّل فك الترميز باستخدام CUDA graphs.

## استنتاج WebUI

### Awesome WebUI

Awesome WebUI هي واجهة ويب حديثة مبنية باستخدام TypeScript، وتوفر ميزات أغنى وتجربة استخدام أفضل.

صور Docker الخاصة بالخادم تتضمن واجهة WebUI المبنية بالفعل: افتح `http://localhost:8080/ui` بعد تشغيل الحاوية. الخطوات التالية تنطبق على التشغيل من المصدر.

**بناء WebUI:**

تحتاج إلى تثبيت Node.js و npm على جهازك أو خادمك.

1. ادخل إلى مجلد `awesome_webui`:
   ```bash
   cd awesome_webui
   ```
2. ثبّت الاعتماديات:
   ```bash
   npm install
   ```
3. ابنِ WebUI:
   ```bash
   npm run build
   ```

**تشغيل الخادم الخلفي:**

بعد بناء WebUI، عد إلى المجلد الجذري للمشروع وشغّل خادم API (`--compile` مخصص لـ S2-Pro فقط، فاحذفه عند تحميل Drama 3):

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**الوصول:**

بعد تشغيل الخادم، يمكنك الوصول إليه عبر المتصفح من العنوان التالي:
`http://localhost:8888/ui`
