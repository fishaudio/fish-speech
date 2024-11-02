# How To Start?

### Download Model

You can get the model by:

```bash
huggingface-cli download fishaudio/fish-agent-v0.1-3b --local-dir checkpoints/fish-agent-v0.1-3b
```

Put them in the 'checkpoints' folder.

You also need the VQGAN weight in the fish-speech-1.4 repo.

So there will be 2 folder in the checkpoints.

The ``checkpoints/fish-speech-1.4`` and ``checkpoints/fish-agent-v0.1-3b``

### Environment Prepare

If you haven't install the environment of Fish-speech, please use:

```bash
pip install -e .[stable]
```

### Launch The Agent Demo.

Please use the command below under the main folder:

```bash
python -m tools.api --llama-checkpoint-path checkpoints/fish-agent-v0.1-3b/ --mode agent --compile
```

The ``--compile`` args only support Python < 3.12 , which will greatly speed up the token generation.

It won't compile at once (remember).

Then please use the command:

```bash
python -m tools.e2e_webui
```

This will create a Gradio WebUI on the device.

When you first use the model, it will come to compile (if the ``--compile`` is True) for a short time, so please wait with patience.

Have a good time!

# About Agent

This model is currently undergoing testing. We welcome suggestions and assistance in improving it.

We are considering refining the tutorial and incorporating it into the main documentation after the testing phase is complete.
