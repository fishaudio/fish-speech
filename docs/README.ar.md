<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **العربية** <br>

<a href="https://www.producthunt.com/posts/fish-speech-1-4?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-fish&#0045;speech&#0045;1&#0045;4" target="_blank">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=488440&theme=light" alt="Fish&#0032;Speech&#0032;1&#0046;4 - Open&#0045;Source&#0032;Multilingual&#0032;Text&#0045;to&#0045;Speech&#0032;with&#0032;Voice&#0032;Cloning | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" />
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
> يتم إصدار قاعدة الكود هذه تحت **رخصة Apache** ويتم إصدار جميع أوزان النماذج تحت **رخصة CC-BY-NC-SA-4.0**. يرجى الرجوع إلى [LICENSE](../LICENSE) لمزيد من التفاصيل.

> [!WARNING]
> **إخلاء المسؤولية القانونية**  
> نحن لا نتحمل أي مسؤولية عن أي استخدام غير قانوني لقاعدة الكود. يرجى الرجوع إلى القوانين المحلية حول DMCA والقوانين الأخرى ذات الصلة.

## ابدأ من هنا

هنا هي الوثائق الرسمية لـ Fish Speech، اتبع التعليمات للبدء بسهولة.

- [التثبيت](https://speech.fish.audio/ar/install/)
- [الاستنتاج](https://speech.fish.audio/ar/inference/)
- [العينات](https://speech.fish.audio/samples)

## 🎉 الإعلان

نحن متحمسون للإعلان عن إعادة تسمية علامتنا التجارية إلى **OpenAudio** — تقديم سلسلة جديدة ثورية من نماذج تحويل النص إلى كلام المتقدمة التي تبني على أساس Fish-Speech.

نحن فخورون بإطلاق **OpenAudio-S1** كأول نموذج في هذه السلسلة، يقدم تحسينات كبيرة في الجودة والأداء والقدرات.

يأتي OpenAudio-S1 في إصدارين: **OpenAudio-S1** و **OpenAudio-S1-mini**. كلا النموذجين متاحان الآن على [Fish Audio Playground](https://fish.audio) (لـ **OpenAudio-S1**) و [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini) (لـ **OpenAudio-S1-mini**).

قم بزيارة [موقع OpenAudio](https://openaudio.com/blogs/s1) للمدونة والتقرير التقني.

## النقاط البارزة ✨

### **جودة TTS ممتازة**

نستخدم مقاييس تقييم Seed TTS لتقييم أداء النموذج، وتظهر النتائج أن OpenAudio S1 يحقق **0.008 WER** و **0.004 CER** على النص الإنجليزي، وهو أفضل بكثير من النماذج السابقة. (الإنجليزية، التقييم التلقائي، بناءً على OpenAI gpt-4o-transcribe، مسافة المتحدث باستخدام Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| النموذج | معدل خطأ الكلمات (WER) | معدل خطأ الأحرف (CER) | مسافة المتحدث |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **أفضل نموذج في TTS-Arena2** 🏆

حقق OpenAudio S1 **المرتبة الأولى** في [TTS-Arena2](https://arena.speechcolab.org/)، المعيار لتقييم تحويل النص إلى كلام:

<div align="center">
    <img src="assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **التحكم في الكلام**
يدعم OpenAudio S1 **مجموعة متنوعة من العلامات العاطفية والنبرة والخاصة** لتعزيز تركيب الكلام:

- **المشاعر الأساسية**:
```
(غاضب) (حزين) (متحمس) (مندهش) (راضي) (مسرور) 
(خائف) (قلق) (منزعج) (عصبي) (محبط) (مكتئب)
(متعاطف) (محرج) (مشمئز) (متأثر) (فخور) (مسترخي)
(ممتن) (واثق) (مهتم) (فضولي) (مرتبك) (مبتهج)
```

- **المشاعر المتقدمة**:
```
(محتقر) (غير سعيد) (قلق) (هستيري) (غير مبال) 
(نافد الصبر) (مذنب) (ازدرائي) (مذعور) (غاضب) (مترد)
(متحمس) (غير موافق) (سلبي) (منكر) (مندهش) (جدي)
(ساخر) (مصالح) (مواسي) (صادق) (ساخر)
(متردد) (مستسلم) (مؤلم) (محرج) (مسلي)
```

- **علامات النبرة**:
```
(بنبرة مستعجلة) (صراخ) (صراخ) (همس) (نبرة ناعمة)
```

- **تأثيرات صوتية خاصة**:
```
(ضحك) (قهقهة) (نشيج) (بكاء بصوت عالٍ) (تنهد) (لهاث)
(أنين) (ضحك الجمهور) (ضحك الخلفية) (ضحك الجمهور)
```

يمكنك أيضًا استخدام ها،ها،ها للتحكم، هناك العديد من الحالات الأخرى في انتظار استكشافك بنفسك.

(الدعم للإنجليزية والصينية واليابانية الآن، والمزيد من اللغات قادم قريبًا!)

### **نوعان من النماذج**

| النموذج | الحجم | التوفر | الميزات |
|-------|------|--------------|----------|
| **S1** | 4 مليار معامل | متاح على [fish.audio](https://fish.audio) | النموذج الرئيسي كامل الميزات |
| **S1-mini** | 0.5 مليار معامل | متاح على Hugging Face [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | إصدار مقطر بالقدرات الأساسية |

كل من S1 و S1-mini يدمجان التعلم المعزز عبر الإنترنت من ردود الفعل البشرية (RLHF).

## **الميزات**

1. **TTS بدون عينات وبعينات قليلة:** أدخل عينة صوتية من 10 إلى 30 ثانية لإنتاج مخرجات TTS عالية الجودة. **للإرشادات التفصيلية، راجع [أفضل ممارسات استنساخ الصوت](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **الدعم متعدد اللغات وعبر اللغات:** ببساطة انسخ والصق النص متعدد اللغات في مربع الإدخال—لا حاجة للقلق بشأن اللغة. يدعم حاليًا الإنجليزية واليابانية والكورية والصينية والفرنسية والألمانية والعربية والإسبانية.

3. **لا يعتمد على الصوتيات:** النموذج لديه قدرات تعميم قوية ولا يعتمد على الصوتيات لـ TTS. يمكنه التعامل مع النص في أي نص لغوي.

4. **دقيق للغاية:** يحقق معدل خطأ أحرف منخفض (CER) حوالي 0.4% ومعدل خطأ كلمات (WER) حوالي 0.8% لـ Seed-TTS Eval.

5. **سريع:** مع تسريع fish-tech، عامل الوقت الحقيقي حوالي 1:5 على كمبيوتر محمول Nvidia RTX 4060 و 1:15 على Nvidia RTX 4090.

6. **استنتاج WebUI:** يتميز بواجهة ويب سهلة الاستخدام قائمة على Gradio متوافقة مع Chrome وFirefox وEdge والمتصفحات الأخرى.

7. **استنتاج GUI:** يوفر واجهة رسومية PyQt6 تعمل بسلاسة مع خادم API. يدعم Linux وWindows وmacOS. [راجع GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **صديق للنشر:** قم بإعداد خادم استنتاج بسهولة مع دعم أصلي لـ Linux وWindows (MacOS قادم قريبًا)، مما يقلل من فقدان السرعة.

## **الوسائط والعروض التوضيحية**

<div align="center">

### **وسائل التواصل الاجتماعي**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="أحدث عرض توضيحي على X" />
</a>

### **العروض التوضيحية التفاعلية**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="جرب OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="جرب S1 Mini" />
</a>

### **عروض الفيديو**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="../docs/assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

### **عينات الصوت**
<div style="margin: 20px 0;">
    <em>ستتوفر عينات صوتية عالية الجودة قريبًا، تُظهر قدراتنا في TTS متعدد اللغات عبر لغات ومشاعر مختلفة.</em>
</div>

</div>

---

## الوثائق

- [بناء البيئة](ar/install.md)
- [الاستنتاج](ar/inference.md)

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
