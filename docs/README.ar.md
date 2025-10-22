<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **العربية** <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2">
      <img alt="TTS-Arena2 Score" src="https://img.shields.io/badge/TTS_Arena2-Rank_%231-gold?style=flat-square&logo=trophy&logoColor=white">
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/openaudio-s1-mini">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **إشعار الترخيص**  
> تم إصدار قاعدة الكود هذه بموجب **ترخيص Apache** وتم إصدار جميع أوزان النموذج بموجب **ترخيص CC-BY-NC-SA-4.0**. يرجى الرجوع إلى [LICENSE](LICENSE) لمزيد من التفاصيل.

> [!WARNING]
> **إخلاء المسؤولية القانوني**  
> نحن لا نتحمل أي مسؤولية عن أي استخدام غير قانوني لقاعدة الكود. يرجى الرجوع إلى القوانين المحلية الخاصة بك فيما يتعلق بقانون الألفية الجديدة لحقوق طبع ونشر المواد الرقمية والقوانين الأخرى ذات الصلة.

## ابدأ هنا

فيما يلي المستندات الرسمية لـ Fish Speech، اتبع التعليمات للبدء بسهولة.

- [التثبيت](https://speech.fish.audio/install/)
- [الضبط الدقيق](https://speech.fish.audio/finetune/)
- [الاستدلال](https://speech.fish.audio/inference/)
- [العينات](https://speech.fish.audio/examples)

## 🎉 إعلان

يسعدنا أن نعلن أننا قمنا بإعادة تسمية العلامة التجارية إلى **OpenAudio** — تقديم سلسلة جديدة ثورية من نماذج تحويل النص إلى كلام المتقدمة التي تبني على أساس Fish-Speech.

نحن فخورون بإصدار **OpenAudio-S1** كنموذج أول في هذه السلسلة، حيث يوفر تحسينات كبيرة في الجودة والأداء والقدرات.

يأتي OpenAudio-S1 في نسختين: **OpenAudio-S1** و **OpenAudio-S1-mini**. كلا النموذجين متاحان الآن على [Fish Audio Playground](https://fish.audio) (لـ **OpenAudio-S1**) و [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini) (لـ **OpenAudio-S1-mini**).

قم بزيارة [موقع OpenAudio](https://openaudio.com/blogs/s1) للمدونة والتقرير التقني.

## أبرز المميزات ✨

### **جودة TTS ممتازة**

نستخدم مقاييس تقييم Seed TTS لتقييم أداء النموذج، وتظهر النتائج أن OpenAudio S1 يحقق **0.008 WER** و **0.004 CER** على النص الإنجليزي، وهو أفضل بشكل ملحوظ من النماذج السابقة. (الإنجليزية، التقييم التلقائي، بناءً على OpenAI gpt-4o-transcribe، مسافة المتحدث باستخدام Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| النموذج | معدل الخطأ في الكلمات (WER) | معدل الخطأ في الأحرف (CER) | مسافة المتحدث |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **أفضل نموذج في TTS-Arena2** 🏆

حقق OpenAudio S1 **المركز الأول** على [TTS-Arena2](https://arena.speechcolab.org/)، المعيار لتقييم تحويل النص إلى كلام:

<div align="center">
    <img src="../docs/assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **التحكم في الكلام**

يدعم OpenAudio S1 **مجموعة متنوعة من العلامات العاطفية والنبرة والعلامات الخاصة** لتعزيز تخليق الكلام:

- **العواطف الأساسية**:
```
(غاضب) (حزين) (متحمس) (مندهش) (راضي) (مسرور) 
(خائف) (قلق) (منزعج) (متوتر) (محبط) (مكتئب)
(متعاطف) (محرج) (مشمئز) (متحرك) (فخور) (مرتاح)
(ممتن) (واثق) (مهتم) (فضولي) (مرتبك) (مبتهج)
```

- **العواطف المتقدمة**:
```
(محتقر) (غير سعيد) (قلق) (هستيري) (غير مبال) 
(غير صبور) (مذنب) (ساخر) (ذعر) (غاضب) (متردد)
(متحمس) (غير موافق) (سلبي) (نافي) (مندهش) (جاد)
(ساخر) (مصالح) (مريح) (صادق) (ساخر)
(متردد) (مستسلم) (مؤلم) (محرج) (مسلي)
```

- **علامات النبرة**:
```
(بنبرة مستعجلة) (يصرخ) (يصرخ) (يهمهم) (بنبرة ناعمة)
```

- **تأثيرات صوتية خاصة**:
```
(يضحك) (يقهقه) (ينتحب) (يبكي بصوت عال) (يتنهد) (يلهث)
(يئن) (ضحك الجمهور) (ضحك في الخلفية) (ضحك الجمهور)
```

يمكنك أيضًا استخدام Ha,ha,ha للتحكم، وهناك العديد من الحالات الأخرى التي تنتظر استكشافها بنفسك.

(الدعم متاح للإنجليزية والصينية واليابانية الآن، والمزيد من اللغات قريبًا!)

### **نوعان من النماذج**

| النموذج | الحجم | التوفر | الميزات |
|-------|------|--------------|----------|
| **S1** | 4B معامل | متوفر على [fish.audio](https://fish.audio/) | النموذج الرئيسي كامل الميزات |
| **S1-mini** | 0.5B معامل | متوفر على huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | نسخة مقطرة بالقدرات الأساسية |

كلا النموذجين S1 و S1-mini يتضمنان التعلم المعزز من التغذية الراجعة البشرية (RLHF) عبر الإنترنت.

## **الميزات**

1. **TTS بدون عينات وقليل العينات:** أدخل عينة صوتية مدتها 10 إلى 30 ثانية لتوليد مخرجات TTS عالية الجودة. **للحصول على إرشادات مفصلة، راجع [أفضل ممارسات استنساخ الصوت](https://docs.fish.audio/resources/best-practices/voice-cloning).**

2. **الدعم متعدد اللغات وعبر اللغات:** ما عليك سوى نسخ ولصق النص متعدد اللغات في مربع الإدخال — لا داعي للقلق بشأن اللغة. يدعم حاليًا الإنجليزية واليابانية والكورية والصينية والفرنسية والألمانية والعربية والإسبانية.

3. **لا يعتمد على الفونيمات:** يتمتع النموذج بقدرات تعميم قوية ولا يعتمد على الفونيمات لـ TTS. يمكنه التعامل مع النص بأي لغة نصية.

4. **دقيق للغاية:** يحقق معدل خطأ في الأحرف (CER) حوالي 0.4٪ ومعدل خطأ في الكلمات (WER) حوالي 0.8٪ لـ Seed-TTS Eval.

5. **سريع:** مع التسريع بواسطة torch compile، فإن عامل الوقت الحقيقي هو حوالي 1:7 على بطاقة Nvidia RTX 4090 GPU.

6. **استدلال WebUI:** يتميز بواجهة ويب سهلة الاستخدام تعتمد على Gradio متوافقة مع Chrome و Firefox و Edge والمتصفحات الأخرى.

7. **سهولة النشر:** يمكنك إعداد خادم استدلال بسهولة مع دعم أصلي لأنظمة Linux و Windows (دعم macOS قريبًا)، مما يقلل من فقدان الأداء.

## **وسائل الإعلام والعروض التوضيحية**

<div align="center">

### **وسائل التواصل الاجتماعي**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-أحدث_عرض_توضيحي-black?style=for-the-badge&logo=x&logoColor=white" alt="أحدث عرض توضيحي على X" />
</a>

### **العروض التوضيحية التفاعلية**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-جرب_OpenAudio_S1-blue?style=for-the-badge" alt="جرب OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-جرب_S1_Mini-yellow?style=for-the-badge" alt="جرب S1 Mini" />
</a>

### **عروض الفيديو**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="../docs/assets/Thumbnail.jpg" alt="فيديو OpenAudio S1" style="width: 50%;" />
</a>

</div>

---

## الاعتمادات

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## التقرير التقني (V1.4)
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}
```
