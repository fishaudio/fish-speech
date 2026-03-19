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

## استنتاج WebUI

### 1. Gradio WebUI

للحفاظ على التوافق، ما زلنا نحتفظ بواجهة Gradio WebUI السابقة.

```bash
python tools/run_webui.py # --compile إذا كنت بحاجة إلى تسريع
```

### 2. Awesome WebUI

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
