## المتطلبات

| النموذج | ذاكرة GPU | GPU |
| --- | --- | --- |
| Drama 3 | 32 GB على الأقل | NVIDIA GPU فقط: لا يدعم CPU أو Apple MPS أو AMD ROCm، ولا يدعم `--compile` |
| S2-Pro | 24 GB على الأقل | NVIDIA GPU أو CPU |

- النظام: Linux، WSL

!!! note
    ذروة استهلاك الذاكرة المقاسة لـ Drama 3 على بطاقة H20 واحدة تبلغ نحو 41 GB، لذا اترك بعض الهامش. بطاقات مثل V100 والأقدم من SM80 لا تدعم bfloat16، فيتحول Drama 3 إلى float16 (مع إبقاء التيار المتبقي fp32).

## إعداد النظام

يتشارك Fish Audio Drama 3 و S2-Pro في نفس طريقة التثبيت. اختر الخيار الأنسب لبيئة التطوير لديك.

**المتطلبات المسبقة**: ثبّت اعتماديات النظام اللازمة لمعالجة الصوت:
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# تثبيت نسخة GPU (اختر إصدار CUDA الخاص بك: cu126، cu128، cu129)
# استبدل كلتا قيمتي "cu129" عند استخدام cu126 أو cu128.
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# تثبيت نسخة CPU فقط
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

# التثبيت الافتراضي (يستخدم فهرس PyTorch الافتراضي)
pip install -e .

# إذا واجهت خطأ أثناء التثبيت بسبب pyaudio، ففكر في استخدام الأمر التالي:
# conda install pyaudio
# ثم قم بتشغيل pip install -e . مرة أخرى
```

### UV

يوفر UV حلًّا وتثبيتًا أسرع للاعتماديات:

```bash
# تثبيت نسخة GPU (اختر إصدار CUDA الخاص بك: cu126، cu128، cu129)
uv sync --python 3.12 --extra cu129

# تثبيت نسخة CPU فقط
uv sync --python 3.12 --extra cpu
```

### دعم Intel Arc XPU

إذا كنت تستخدم بطاقة Intel Arc، فثبّت مع دعم XPU (لـ S2-Pro فقط؛ Drama 3 يحتاج بطاقة NVIDIA):

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# تثبيت مكتبة C++ القياسية المطلوبة
conda install libstdcxx -c conda-forge

# تثبيت PyTorch مع دعم Intel XPU
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# تثبيت Fish Speech
pip install -e .
```

!!! warning
    خيار `compile` غير مدعوم على Windows و macOS. إذا أردت التشغيل مع compile، عليك تثبيت Triton يدويًا.

## تحميل الأوزان

يتشارك النموذجان في نفس سطر الأوامر وخادم API، ويُستنتج نوع النموذج من ملف `config.json` في مجلد الأوزان. ضعهما في نفس مجلد `checkpoints`.

التحميل يحتاج أداة سطر أوامر Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

إذا لم يكن ملف `codec.pth` موجودًا في مجلد الأوزان، فسيُستخدم codec الخاص بـ S2-Pro من `checkpoints/s2-pro/codec.pth`، لذا يُنصح بتحميل S2-Pro أيضًا.

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

بعد تحميل الأوزان، راجع [الاستنتاج](inference.md).

## إعداد Docker

يوفر Drama 3 و S2-Pro عدة طرق للنشر باستخدام Docker. يمكنك استخدام صور جاهزة من Docker Hub، أو البناء محليًا باستخدام Docker Compose، أو بناء صور مخصصة يدويًا.

نوفر صورًا لخادم API تعمل على GPU (CUDA126 افتراضيًا) وعلى CPU. الصور تتضمن واجهة WebUI بالفعل: افتح `http://localhost:8080/ui` بعد تشغيل الحاوية. صورة GPU قادرة على تشغيل النموذجين، أما صورة CPU فتدعم S2-Pro فقط.

### المتطلبات الأساسية

- تثبيت Docker و Docker Compose
- (لـ GPU) تثبيت NVIDIA Docker runtime (إلزامي لـ Drama 3)
- ذاكرة GPU: 32 GB على الأقل لـ Drama 3، ويفضّل 24 GB لـ S2-Pro

### تشغيل الصورة الجاهزة

Drama 3: على البطاقات SM89 أو الأحدث، اكشف بطاقة واحدة فقط كما يلي؛ أما البطاقات الأقدم فاستخدم `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

S2-Pro:

```bash
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### استخدام Docker Compose

إذا كنت تحتاج إلى التطوير أو التخصيص، فننصح باستخدام Docker Compose للبناء والتشغيل محليًا:

```bash
# أولاً، استنسخ المستودع
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# بدء خادم API (يحمّل checkpoints/s2-pro افتراضيًا)
docker compose --profile server up

# تحميل Drama 3 وتثبيته على بطاقة واحدة عبر DEVICE (يصل إلى أربع بطاقات عند تركه فارغًا)
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# بدء خادم API مع تحسين التجميع (لـ S2-Pro فقط)
COMPILE=1 docker compose --profile server up

# النشر باستخدام CPU فقط (لـ S2-Pro فقط)
BACKEND=cpu docker compose --profile server up
```

#### متغيرات البيئة لـ Docker Compose

يمكنك تخصيص النشر باستخدام متغيرات البيئة:

```bash
# مثال على ملف .env
BACKEND=cuda              # أو cpu
COMPILE=1                 # تمكين تحسين التجميع
API_PORT=8080             # منفذ خادم API
UV_VERSION=0.8.15         # إصدار مدير الحزم UV
```

سينفّذ الأمر بناء الصورة وتشغيل الحاوية. يمكنك الوصول إلى خادم API عبر `http://localhost:8080` وإلى WebUI عبر `http://localhost:8080/ui`.

### البناء اليدوي باستخدام Docker

لمن يريد تحكمًا أدق في عملية البناء:

```bash
# بناء صورة خادم API مع دعم CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# بناء صورة CPU فقط (تدعم منصات متعددة)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target server \
    -t fish-speech-server:cpu .

# بناء صورة التطوير
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### وسيطات البناء

- `BACKEND`: `cuda` أو `cpu` (الافتراضي: `cuda`)
- `CUDA_VER`: إصدار CUDA (الافتراضي: `12.6.0`)
- `UV_EXTRA`: إضافة UV الخاصة بـ CUDA (الافتراضي: `cu126`)
- `UBUNTU_VER`: إصدار Ubuntu (الافتراضي: `24.04`)
- `PY_VER`: إصدار Python (الافتراضي: `3.12`)

### تحميل المجلدات

تحتاج الطريقتان إلى تحميل المجلدات التالية:

- `./checkpoints:/app/checkpoints` - مجلد أوزان النموذج
- `./references:/app/references` - مجلد ملفات الصوت المرجعية

### متغيرات البيئة

- `COMPILE=1` - تمكين torch.compile لاستنتاج أسرع (حوالي 10 أضعاف)
- `LLAMA_CHECKPOINT_PATH` - مسار أوزان النموذج الرئيسي، مثل `checkpoints/drama-3` أو `checkpoints/s2-pro`
- `DECODER_CHECKPOINT_PATH` - مسار أوزان الـ codec؛ إذا تُرك فارغًا فسيحدده الخادم من مجلد الأوزان
- `DEVICE` - جهاز الاستنتاج، مثل `cuda:0`؛ يستخدم Drama 3 تلك البطاقة فقط عند تحديدها، وما يصل إلى أربع بطاقات عند تركه فارغًا
- `FISH_DRAMA3_FP8_EXPERTS=0` أو `1` - يفرض على Drama 3 استخدام خبراء غير مكمَّمين أو بتنسيق FP8 (يُختار تلقائيًا حسب البطاقة افتراضيًا)
- `FISH_DRAMA3_CUDA_GRAPH=0` - يعطّل فك الترميز باستخدام CUDA graphs في Drama 3
- `API_SERVER_NAME=0.0.0.0` - عنوان خادم API
- `API_SERVER_PORT=8080` - منفذ خادم API

!!! note
    حاويات Docker تقرأ أوزان النماذج من `/app/checkpoints`. تأكد من تحميل الأوزان المطلوبة قبل تشغيل الحاويات.

!!! warning
    دعم GPU يتطلب NVIDIA Docker runtime. عند النشر باستخدام CPU فقط، احذف `--gpus all` واستخدم صور CPU.
