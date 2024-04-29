## Functions of fish-speech
fish-speech is a pre-trained TTS model mainly used to generate speech with ACGN style. It requires only a few tens of seconds of speech data to generate more realistic character speech. We provide a 1B-parameter pre-trained model, you can deploy the model locally to make your favorite characters generate the words you want.

We also support terminal command line, http api interface, and webui for inferencing. Traveler, Pathfinder, Captain, Doctor, sensei, Demon Hunter, Meow Meow Lou, and V can choose their preferred way to do inferencing.

Considering that Traveler, Pathfinder, Captain, Doctor, sensei, Demon Hunter, Meow Meow Lou, and V who use this project for TTS tasks may not be familiar with computers and AI modeling related content, we have also prepared a detailed background knowledge and multi-platform installation guide in the hope that Traveler, Pathfinder, Captain, Doctor, sensei, Demon Hunter, Meow Meow Lou, and V will be able to generate their own preferred speech.

## What are the differences and advantages of fish-speech compared to Bert-vits2
fish-speech is a new autoregressive TTS model that achieves better speech generation, more language support, longer individual speech generation, and faster generation speed compared to Bert-vits2.

## Frameworks of our project
This project uses a speech generation architecture similar to VALL-E to generate quantized Mel tokens using a fine-tuned specialization of Llama as a base for the macromodel.Afterwards, they are converted to real audio by a vocoder with the VQ-GAN architecture.Please refer to the Architecture section for the specific project architecture.

## Model finetuning
If you are not satisfied with the results of our pre-training model, we provide Lora and SFT based fine-tuning functions, you can add your own dataset to fine-tune our pre-training model.

**Enjoy!**
