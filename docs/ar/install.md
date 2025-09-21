## المتطلبات

- ذاكرة وحدة معالجة الرسومات (GPU): 12 جيجابايت (للاستدلال)
- النظام: Linux, WSL

## إعداد النظام

يدعم OpenAudio طرق تثبيت متعددة. اختر الطريقة التي تناسب بيئة التطوير الخاصة بك.

**المتطلبات الأساسية**: قم بتثبيت تبعيات النظام لمعالجة الصوت:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# تثبيت نسخة GPU (اختر إصدار CUDA الخاص بك: cu126, cu128, cu129)
pip install -e .[cu129]

# تثبيت نسخة CPU فقط
pip install -e .[cpu]

# التثبيت الافتراضي (يستخدم فهرس PyTorch الافتراضي)
pip install -e .
```

### UV

يوفر UV حلاً أسرع لتثبيت التبعيات:

```bash
# تثبيت نسخة GPU (اختر إصدار CUDA الخاص بك: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# تثبيت نسخة CPU فقط
uv sync --python 3.12 --extra cpu
```
### دعم Intel Arc XPU

لمستخدمي وحدات معالجة الرسومات Intel Arc، قم بالتثبيت مع دعم XPU على النحو التالي:

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
    خيار `compile` غير مدعوم على أنظمة Windows و macOS. إذا كنت ترغب في التشغيل مع التجميع، ستحتاج إلى تثبيت Triton بنفسك.


## إعداد Docker

يوفر نموذج سلسلة OpenAudio S1 خيارات نشر متعددة مع Docker لتلبية الاحتياجات المختلفة. يمكنك استخدام الصور المعدة مسبقًا من Docker Hub، أو البناء محليًا باستخدام Docker Compose، أو بناء صور مخصصة يدويًا.

لقد قدمنا صور Docker لكل من واجهة المستخدم الرسومية (WebUI) وخادم API، لكل من وحدات معالجة الرسومات (GPU) (CUDA 12.6 افتراضيًا) ووحدات المعالجة المركزية (CPU). يمكنك استخدام الصور المعدة مسبقًا من Docker Hub، أو البناء محليًا باستخدام Docker Compose، أو بناء صور مخصصة يدويًا. إذا كنت ترغب في البناء محليًا، فاتبع الإرشادات أدناه. إذا كنت ترغب فقط في استخدام الصور المعدة مسبقًا، فاتبع مباشرةً [دليل الاستدلال](inference.md).

### المتطلبات الأساسية

- تثبيت Docker و Docker Compose
- تثبيت NVIDIA Docker runtime (لدعم GPU)
- ذاكرة GPU لا تقل عن 12 جيجابايت للاستدلال باستخدام CUDA

### استخدام Docker Compose

للتطوير أو التخصيص، يمكنك استخدام Docker Compose للبناء والتشغيل محليًا:

```bash
# أولاً، استنسخ المستودع
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# بدء واجهة المستخدم الرسومية (WebUI) مع CUDA
docker compose --profile webui up

# بدء واجهة المستخدم الرسومية (WebUI) مع تحسين التجميع
COMPILE=1 docker compose --profile webui up

# بدء خادم API
docker compose --profile server up

# بدء خادم API مع تحسين التجميع
COMPILE=1 docker compose --profile server up

# النشر باستخدام CPU فقط
BACKEND=cpu docker compose --profile webui up
```

#### متغيرات البيئة لـ Docker Compose

يمكنك تخصيص النشر باستخدام متغيرات البيئة:

```bash
# مثال على ملف .env
BACKEND=cuda              # أو cpu
COMPILE=1                 # تمكين تحسين التجميع
GRADIO_PORT=7860         # منفذ واجهة المستخدم الرسومية (WebUI)
API_PORT=8080            # منفذ خادم API
UV_VERSION=0.8.15        # إصدار مدير الحزم UV
```

سيقوم الأمر ببناء الصورة وتشغيل الحاوية. يمكنك الوصول إلى واجهة المستخدم الرسومية (WebUI) على `http://localhost:7860` وخادم API على `http://localhost:8080`.

### البناء اليدوي باستخدام Docker

للمستخدمين المتقدمين الذين يرغبون في تخصيص عملية البناء:

```bash
# بناء صورة واجهة المستخدم الرسومية (WebUI) مع دعم CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

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
    --target webui \
    -t fish-speech-webui:cpu .

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
- `UV_EXTRA`: حزمة UV إضافية لـ CUDA (الافتراضي: `cu126`)
- `UBUNTU_VER`: إصدار Ubuntu (الافتراضي: `24.04`)
- `PY_VER`: إصدار Python (الافتراضي: `3.12`)

### تحميل المجلدات

تتطلب كلتا الطريقتين تحميل المجلدات التالية:

- `./checkpoints:/app/checkpoints` - مجلد أوزان النموذج
- `./references:/app/references` - مجلد ملفات الصوت المرجعية

### متغيرات البيئة

- `COMPILE=1` - تمكين `torch.compile` لتسريع الاستدلال (حوالي 10 أضعاف)
- `GRADIO_SERVER_NAME=0.0.0.0` - مضيف خادم واجهة المستخدم الرسومية (WebUI)
- `GRADIO_SERVER_PORT=7860` - منفذ خادم واجهة المستخدم الرسومية (WebUI)
- `API_SERVER_NAME=0.0.0.0` - مضيف خادم API
- `API_SERVER_PORT=8080` - منفذ خادم API

!!! note
    تتوقع حاويات Docker أن يتم تحميل أوزان النموذج في `/app/checkpoints`. تأكد من تنزيل أوزان النموذج المطلوبة قبل بدء الحاويات.

!!! warning
    يتطلب دعم GPU وجود NVIDIA Docker runtime. للنشر باستخدام CPU فقط، قم بإزالة علامة `--gpus all` واستخدم صور CPU.
