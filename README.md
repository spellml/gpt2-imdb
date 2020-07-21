# gpt2-imdb <a href="https://web.spell.ml/workspace_create?workspaceName=gpt2-imdb&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fgpt2-imdb&pip=transformers,tokenizers,nlp"><img src=https://spell.ml/badge.svg height=20px/></a>

`pytorch` scripts training GPT2 (trained from scratch) on the IMDB movie reviews dataset. The model and data are via `hugginface` `transformers`.

Even with the standard settings GPT2 is such a large model that it's hard to fit many items per batch onto a single video card! That makes this training script useful for benchmarking a large-scale attention NLP model training job.

To run on Spell:

```bash
spell run --machine-type v100 \
  --github-url https://github.com/ResidentMario/spell-gpt2-imdb.git \
  --pip transformers --pip nlp --pip tokenizers \
  --tensorboard-dir /spell/tensorboards/model_1/ \
  "python models/model_1.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/ResidentMario/spell-gpt2-imdb.git \
  --pip transformers --pip nlp --pip tokenizers \
  --tensorboard-dir /spell/tensorboards/model_2/ \
  "python models/model_2.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/ResidentMario/spell-gpt2-imdb.git \
  --pip transformers --pip nlp --pip tokenizers \
  --tensorboard-dir /spell/tensorboards/model_2/ \
  "python models/model_2.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/ResidentMario/spell-gpt2-imdb.git \
  --pip transformers --pip nlp --pip tokenizers \
  --tensorboard-dir /spell/tensorboards/model_3/ \
  "python models/model_3.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/ResidentMario/spell-gpt2-imdb.git \
  --pip transformers --pip nlp --pip tokenizers \
  --tensorboard-dir /spell/tensorboards/model_3/ \
  "python models/model_3.py"
```
