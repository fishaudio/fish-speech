<div align="center">
<h1>Fish Speech</h1>

[English](../en/) | [简体中文](../zh/) | [Portuguese](../pt/) | [日本語](../ja/) | [한국어](../ko/) | **العربية** <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

!!! info "تنبيه الترخيص"
    يتم إصدار قاعدة الأكواد هذه بموجب **ترخيص Apache** ويتم إصدار جميع أوزان النماذج بموجب **ترخيص CC-BY-NC-SA-4.0**. يرجى الرجوع إلى [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) لمزيد من التفاصيل.

!!! warning "إخلاء المسؤولية القانونية"
    نحن لا نتحمل أي مسؤولية عن أي استخدام غير قانوني لقاعدة الأكواد. يرجى مراجعة القوانين المحلية المتعلقة بـ DMCA والقوانين الأخرى ذات الصلة.

## ابدأ من هنا

هذا هو الوثائق الرسمية لـ Fish Speech. يرجى اتباع التعليمات للبدء بسهولة.

- [التثبيت](install.md)
- [الاستنتاج](inference.md)

## Fish Audio S2
**أفضل نظام لتحويل النص إلى كلام في كل من المصادر المفتوحة والمغلقة**

Fish Audio S2 هو أحدث نموذج تم تطويره بواسطة [Fish Audio](https://fish.audio/)، وهو مصمم لتوليد كلام يبدو طبيعيًا وأصليًا وغنيًا بالعاطفة — غير ميكانيكي أو مسطح أو مقتصر على القراءة بأسلوب الاستوديو.

يركز Fish Audio S2 على المحادثات اليومية، ويدعم توليد المتحدثين المتعددين الأصليين وتوليد الحوارات متعددة الأدوار. كما يدعم التحكم التعليمي.

تتضمن سلسلة S2 نماذج متعددة. النموذج المفتوح المصدر هو S2-Pro، وهو أقوى نموذج في السلسلة.

يرجى زيارة [موقع Fish Audio](https://fish.audio/) لتجربة فورية.

### متغيرات النموذج

| النموذج | الحجم | التوفر | الوصف |
|------|------|-------------|-------------|
| S2-Pro | 4B معاملات | [huggingface](https://huggingface.co/fishaudio/s2-pro) | نموذج رائد بكامل الميزات مع أعلى جودة واستقرار |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | نموذجنا المغلق المصدر بسرعات أعلى وزمن وصول أقل |

لمزيد من التفاصيل حول النماذج ، يرجى مراجعة التقرير الفني.

## أبرز المميزات

<img src="../assets/totalability.png" width=200%>

### التحكم باللغة الطبيعية

يسمح Fish Audio S2 للمستخدمين باستخدام اللغة الطبيعية للتحكم في أداء كل جملة ، والمعلومات غير اللفظية ، والعواطف ، والمزيد من خصائص الصوت ، بدلاً من مجرد استخدام علامات قصيرة للتحكم بشكل غامض في أداء النموذج. يؤدي ذلك إلى تحسين الجودة الإجمالية للمحتوى المولّد بشكل كبير.

### دعم لغات متعددة

يدعم Fish Audio S2 تحويل النص إلى كلام متعدد اللغات بجودة عالية دون الحاجة إلى وحدات صوتية أو معالجة مسبقة خاصة باللغة. يشمل ذلك:

**الإنجليزية ، الصينية ، اليابانية ، الكورية ، العربية ، الألمانية ، الفرنسية ...**

**والمزيد في المستقبل!**

القائمة تتوسع باستمرار ، يرجى التحقق من [Fish Audio](https://fish.audio/) للحصول على أحدث الإصدارات.

### توليد المتحدثين المتعددين الأصليين

<img src="../assets/chattemplate.png" width=200%>

يسمح Fish Audio S2 للمستخدمين بتحميل عينات صوتية مرجعية تحتوي على متحدثين متعددين ، وسيقوم النموذج بمعالجة خصائص كل متحدث من خلال رمز `<|speaker:i|>`. بعد ذلك ، يمكنك التحكم في أداء النموذج عبر رموز معرف المتحدث ، مما يحقق تعدد المتحدثين في عملية توليد واحدة. لا داعي بعد الآن لتحميل أصوات مرجعية وتوليد كلام لكل متحدث على حدة.

### توليد الحوارات متعددة الأدوار

بفضل توسيع سياق النموذج ، يمكن لنموذجنا الآن استخدام معلومات السياق السابق لتحسين التعبير عن المحتوى المولّد لاحقًا ، وبالتالي زيادة طبيعية المحتوى.

### استنساخ الصوت السريع

يدعم Fish Audio S2 استنساخ الصوت الدقيق باستخدام عينات مرجعية قصيرة (عادة 10-30 ثانية). يمكن للنموذج التقاط نبرة الصوت وأسلوب التحدث والميل العاطفي ، وتوليد أصوات مستنسخة واقعية ومتسقة دون ضبط دقيق إضافي.

---

## شكر وتقدير

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## التقرير الفني

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
