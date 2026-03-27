#!/usr/bin/env python3

# Copyright 2023 The AI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates an interactive Deep Learning Roadmap MkDocs Markdown page."""

import pathlib

OUTPUT_PATH = (
  pathlib.Path(__file__).parent.parent / 'docs' / 'deep-learning-roadmap.md'
)

# (category_name, hex_color, [(concept, article_url, video_url), ...])
ROADMAP = [
  (
    'Foundations', '#4A90D9', [
      (
        'Understanding Neural Networks',
        'https://www.3blue1brown.com/lessons/neural-networks',
        'https://www.youtube.com/watch?v=aircAruvnKk',
      ),
      (
        'Loss Functions',
        'https://cs231n.github.io/neural-networks-1/',
        'https://www.youtube.com/watch?v=Skc8nqJirJg',
      ),
      (
        'Activation Functions',
        'https://cs231n.github.io/neural-networks-1/',
        'https://www.youtube.com/watch?v=VMj-3S1tku0',
      ),
      (
        'Weight Initialization',
        'https://cs231n.github.io/neural-networks-2/',
        'https://www.youtube.com/watch?v=P6sfmUTpUmc',
      ),
      (
        'Vanishing / Exploding Gradients',
        'https://cs231n.github.io/neural-networks-3/',
        'https://www.youtube.com/watch?v=Ilg3gGewQ5U',
      ),
    ],
  ),
  (
    'Architectures', '#7B68EE', [
      (
        'Feedforward Neural Network',
        'https://cs231n.github.io/neural-networks-1/',
        'https://www.youtube.com/watch?v=aircAruvnKk',
      ),
      (
        'Autoencoder',
        'https://lilianweng.github.io/posts/2018-08-12-vae/',
        'https://www.youtube.com/watch?v=VMj-3S1tku0',
      ),
      (
        'Convolutional Neural Network',
        'https://cs231n.github.io/convolutional-networks/',
        'https://www.youtube.com/watch?v=NfnWJUyUJYU',
      ),
      (
        'Recurrent Neural Network',
        'https://d2l.ai/chapter_recurrent-neural-networks/index.html',
        'https://www.youtube.com/watch?v=ySEx_Bqxvvo',
      ),
      (
        'Transformer',
        'https://d2l.ai/chapter_attention-mechanisms-and-transformers'
        '/transformer.html',
        'https://www.youtube.com/watch?v=wjZofJX0v4M',
      ),
      (
        'Siamese Network',
        'https://lilianweng.github.io/posts/2021-05-31-contrastive/',
        'https://www.youtube.com/watch?v=6VRFi7Rn6hk',
      ),
      (
        'Generative Adversarial Network',
        'https://distill.pub/2019/gan-open-problems/',
        'https://www.youtube.com/watch?v=dCKbRCUyop8',
      ),
      (
        'Residual Connections',
        'https://d2l.ai/chapter_convolutional-modern/resnet.html',
        'https://www.youtube.com/watch?v=P6sfmUTpUmc',
      ),
      (
        'LSTM',
        'https://d2l.ai/chapter_recurrent-modern/lstm.html',
        'https://www.youtube.com/watch?v=YCzL96nL7j0',
      ),
      (
        'GRU',
        'https://d2l.ai/chapter_recurrent-modern/gru.html',
        'https://www.youtube.com/watch?v=8HyCNIVRbSU',
      ),
      (
        'Encoder / Decoder',
        'https://d2l.ai/chapter_recurrent-modern/seq2seq.html',
        'https://www.youtube.com/watch?v=ySEx_Bqxvvo',
      ),
      (
        'Attention',
        'https://distill.pub/2016/augmented-rnns/',
        'https://www.youtube.com/watch?v=eMlx5fFNoYc',
      ),
    ],
  ),
  (
    'Training', '#2E8B57', [
      (
        'Learning Rate Schedule',
        'https://docs.fast.ai/callback.schedule.html',
        'https://www.youtube.com/watch?v=AcA8HAYh7IE',
      ),
      (
        'Batch Normalization',
        'https://cs231n.github.io/neural-networks-2/',
        'https://www.youtube.com/watch?v=P6sfmUTpUmc',
      ),
      (
        'Batch Size Effects',
        'https://cs231n.github.io/neural-networks-3/',
        'https://www.youtube.com/watch?v=AcA8HAYh7IE',
      ),
      (
        'Multitask Learning',
        'https://lilianweng.github.io/posts/2018-09-24-multitask-learning/',
        'https://www.youtube.com/watch?v=0rZtSwNOTQo',
      ),
      (
        'Transfer Learning',
        'https://course.fast.ai/',
        'https://www.youtube.com/watch?v=htiNBPxcXgo',
      ),
      (
        'Curriculum Learning',
        'https://paperswithcode.com/methods/category/curriculum-learning',
        'https://www.youtube.com/watch?v=W7yOtfeF9OQ',
      ),
    ],
  ),
  (
    'Optimizers', '#E07B00', [
      (
        'SGD',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=sDv4f4s2SB8',
      ),
      (
        'Momentum',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=k8fTYJPd3_I',
      ),
      (
        'Adam',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=JXQT_vxqwIs',
      ),
      (
        'AdaGrad',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=IHZwWFHWa-w',
      ),
      (
        'RMSProp',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=_e-LFe_igno',
      ),
      (
        'AdaDelta',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=IHZwWFHWa-w',
      ),
      (
        'Nadam',
        'https://www.ruder.io/optimizing-gradient-descent/',
        'https://www.youtube.com/watch?v=JXQT_vxqwIs',
      ),
    ],
  ),
  (
    'Regularization', '#C0392B', [
      (
        'Early Stopping',
        'https://cs231n.github.io/neural-networks-3/',
        'https://www.youtube.com/watch?v=UtNsSDfSXgU',
      ),
      (
        'Dropout',
        'https://cs231n.github.io/neural-networks-2/',
        'https://www.youtube.com/watch?v=D8PJAL-MZv8',
      ),
      (
        'Parameter Penalties',
        'https://cs231n.github.io/neural-networks-2/',
        'https://www.youtube.com/watch?v=Q81RR3yKn30',
      ),
      (
        'Data Augmentation',
        'https://course.fast.ai/',
        'https://www.youtube.com/watch?v=htiNBPxcXgo',
      ),
      (
        'Adversarial Training',
        'https://distill.pub/2019/advex-bugs-features/',
        'https://www.youtube.com/watch?v=dOG-HxpbMSY',
      ),
    ],
  ),
  (
    'Model Optimization', '#16A085', [
      (
        'Distillation',
        'https://lilianweng.github.io/posts/2022-09-08-ntk/',
        'https://www.youtube.com/watch?v=rzW0oA-1Ahg',
      ),
      (
        'Quantization',
        'https://huggingface.co/docs/optimum/concept_guides/quantization',
        'https://www.youtube.com/watch?v=DWRl8cpSUdk',
      ),
      (
        'Neural Architecture Search',
        'https://paperswithcode.com/methods/category'
        '/neural-architecture-search',
        'https://www.youtube.com/watch?v=sROrvtXnT7Q',
      ),
    ],
  ),
  (
    'Tools', '#5D6D7E', [
      (
        'PyTorch',
        'https://docs.pytorch.org/tutorials/',
        'https://www.youtube.com/watch?v=EMXfZB8FVUA',
      ),
      (
        'TensorBoard',
        'https://docs.pytorch.org/tutorials/intermediate'
        '/tensorboard_tutorial.html',
        'https://www.youtube.com/watch?v=RLqsxWaQdHE',
      ),
      (
        'MLFlow',
        'https://mlflow.org/docs/latest/',
        'https://www.youtube.com/watch?v=3VFneBfMBJk',
      ),
      (
        'Hugging Face Transformers',
        'https://huggingface.co/docs/transformers/en/index',
        'https://www.youtube.com/watch?v=00GKzGyWFEs',
      ),
    ],
  ),
]


def _cards(concepts: list, color: str) -> str:
  parts = []
  for name, article, video in concepts:
    parts.append(
      f'<div class="roadmap-node" style="--node-color:{color}">'
      f'<span class="name">{name}</span>'
      f'<div class="links">'
      f'<a href="{article}" class="article"'
      f' target="_blank" rel="noopener">article</a>'
      f'<a href="{video}" class="video"'
      f' target="_blank" rel="noopener">&#9654; video</a>'
      f'</div></div>'
    )
  return '\n'.join(parts)


def _sections() -> str:
  parts = []
  for cat_name, color, concepts in ROADMAP:
    parts.append(
      f'\n## {cat_name}\n\n'
      f'<div class="roadmap-grid">\n{_cards(concepts, color)}\n</div>\n'
    )
  return '\n'.join(parts)


def generate() -> str:
  return f'''# Deep Learning Roadmap

PyTorch-focused — click any node to open the article or video.

{_sections()}
'''


# ---------- dead code below kept for historical reference ----------
def _legacy_generate() -> str:
  return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Deep Learning Roadmap</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
      background: #0f0f14;
      color: #e0e0e0;
      min-height: 100vh;
      padding: 2rem 1rem 4rem;
    }}

    h1 {{
      text-align: center;
      font-size: clamp(1.6rem, 4vw, 2.4rem);
      font-weight: 800;
      letter-spacing: 0.04em;
      color: #fff;
      margin-bottom: 0.4rem;
    }}

    .subtitle {{
      text-align: center;
      color: #888;
      font-size: 0.9rem;
      margin-bottom: 3rem;
    }}

    .roadmap {{
      max-width: 1100px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}

    section {{
      border: 1.5px solid color-mix(in srgb, var(--c) 40%, transparent);
      border-radius: 14px;
      padding: 1.25rem 1.5rem 1.5rem;
      background: color-mix(in srgb, var(--c) 6%, #0f0f14);
    }}

    section h2 {{
      font-size: 0.85rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--c);
      margin-bottom: 1rem;
    }}

    .grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}

    .node {{
      background: color-mix(in srgb, var(--c) 12%, #1a1a26);
      border: 1px solid color-mix(in srgb, var(--c) 35%, transparent);
      border-radius: 10px;
      padding: 0.65rem 0.9rem 0.5rem;
      min-width: 160px;
      flex: 1 1 160px;
      max-width: 220px;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      transition: border-color 0.15s, transform 0.15s;
    }}

    .node:hover {{
      border-color: var(--c);
      transform: translateY(-2px);
    }}

    .name {{
      font-size: 0.82rem;
      font-weight: 600;
      color: #e8e8f0;
      line-height: 1.3;
    }}

    .links {{
      display: flex;
      gap: 0.5rem;
    }}

    .links a {{
      font-size: 0.72rem;
      text-decoration: none;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-weight: 500;
      transition: opacity 0.15s;
    }}

    .links a:hover {{ opacity: 0.75; }}

    .links a:not(.yt) {{
      background: color-mix(in srgb, var(--c) 20%, transparent);
      color: var(--c);
    }}

    .links a.yt {{
      background: color-mix(in srgb, #ff4444 20%, transparent);
      color: #ff6b6b;
    }}

    @media (max-width: 600px) {{
      .node {{ max-width: 100%; }}
    }}
  </style>
</head>
<body>
  <h1>Deep Learning Roadmap</h1>
  <p class="subtitle">PyTorch-focused &mdash; click any node to open the article
  or video</p>
  <div class="roadmap">
    {_sections()}
  </div>
</body>
</html>
'''


if __name__ == '__main__':
  md = generate()
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT_PATH.write_text(md, encoding='utf-8')
  print(f'Written to {OUTPUT_PATH}')
