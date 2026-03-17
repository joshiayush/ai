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
"""Generates an interactive Deep Learning Roadmap SVG with clickable links."""

import pathlib
import textwrap

OUTPUT_PATH = (pathlib.Path(__file__).parent.parent / 'docs' / '__design__' /
               'media' / 'Deep_Learning_Roadmap.svg')

# (category_name, hex_color, [(concept, article_url, video_url), ...])
ROADMAP = [
    (
        'Foundations',
        '#4A90D9',
        [
            (
                'Understanding\nNeural Networks',
                'https://www.3blue1brown.com/lessons/neural-networks',
                'https://www.youtube.com/watch?v=aircAruvnKk',
            ),
            (
                'Loss Functions',
                'https://cs231n.github.io/neural-networks-1/',
                'https://www.youtube.com/watch?v=Skc8nqJirJg',
            ),
            (
                'Activation\nFunctions',
                'https://cs231n.github.io/neural-networks-1/',
                'https://www.youtube.com/watch?v=VMj-3S1tku0',
            ),
            (
                'Weight\nInitialization',
                'https://cs231n.github.io/neural-networks-2/',
                'https://www.youtube.com/watch?v=P6sfmUTpUmc',
            ),
            (
                'Vanishing /\nExploding Gradients',
                'https://cs231n.github.io/neural-networks-3/',
                'https://www.youtube.com/watch?v=Ilg3gGewQ5U',
            ),
        ],
    ),
    (
        'Architectures',
        '#7B68EE',
        [
            (
                'Feedforward\nNeural Network',
                'https://cs231n.github.io/neural-networks-1/',
                'https://www.youtube.com/watch?v=aircAruvnKk',
            ),
            (
                'Autoencoder',
                'https://lilianweng.github.io/posts/2018-08-12-vae/',
                'https://www.youtube.com/watch?v=VMj-3S1tku0',
            ),
            (
                'Convolutional\nNeural Network',
                'https://cs231n.github.io/convolutional-networks/',
                'https://www.youtube.com/watch?v=NfnWJUyUJYU',
            ),
            (
                'Recurrent\nNeural Network',
                'https://d2l.ai/chapter_recurrent-neural-networks/index.html',
                'https://www.youtube.com/watch?v=ySEx_Bqxvvo',
            ),
            (
                'Transformer',
                'https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html',
                'https://www.youtube.com/watch?v=wjZofJX0v4M',
            ),
            (
                'Siamese\nNetwork',
                'https://lilianweng.github.io/posts/2021-05-31-contrastive/',
                'https://www.youtube.com/watch?v=6VRFi7Rn6hk',
            ),
            (
                'Generative\nAdversarial Network',
                'https://distill.pub/2019/gan-open-problems/',
                'https://www.youtube.com/watch?v=dCKbRCUyop8',
            ),
            (
                'Residual\nConnections',
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
                'Encoder /\nDecoder',
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
        'Training',
        '#2E8B57',
        [
            (
                'Learning Rate\nSchedule',
                'https://docs.fast.ai/callback.schedule.html',
                'https://www.youtube.com/watch?v=AcA8HAYh7IE',
            ),
            (
                'Batch\nNormalization',
                'https://cs231n.github.io/neural-networks-2/',
                'https://www.youtube.com/watch?v=P6sfmUTpUmc',
            ),
            (
                'Batch Size\nEffects',
                'https://cs231n.github.io/neural-networks-3/',
                'https://www.youtube.com/watch?v=AcA8HAYh7IE',
            ),
            (
                'Multitask\nLearning',
                'https://lilianweng.github.io/posts/2018-09-24-multitask-learning/',
                'https://www.youtube.com/watch?v=0rZtSwNOTQo',
            ),
            (
                'Transfer\nLearning',
                'https://course.fast.ai/',
                'https://www.youtube.com/watch?v=htiNBPxcXgo',
            ),
            (
                'Curriculum\nLearning',
                'https://paperswithcode.com/methods/category/curriculum-learning',
                'https://www.youtube.com/watch?v=W7yOtfeF9OQ',
            ),
        ],
    ),
    (
        'Optimizers',
        '#E07B00',
        [
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
        'Regularization',
        '#C0392B',
        [
            (
                'Early\nStopping',
                'https://cs231n.github.io/neural-networks-3/',
                'https://www.youtube.com/watch?v=UtNsSDfSXgU',
            ),
            (
                'Dropout',
                'https://cs231n.github.io/neural-networks-2/',
                'https://www.youtube.com/watch?v=D8PJAL-MZv8',
            ),
            (
                'Parameter\nPenalties',
                'https://cs231n.github.io/neural-networks-2/',
                'https://www.youtube.com/watch?v=Q81RR3yKn30',
            ),
            (
                'Data\nAugmentation',
                'https://course.fast.ai/',
                'https://www.youtube.com/watch?v=htiNBPxcXgo',
            ),
            (
                'Adversarial\nTraining',
                'https://distill.pub/2019/advex-bugs-features/',
                'https://www.youtube.com/watch?v=dOG-HxpbMSY',
            ),
        ],
    ),
    (
        'Model Optimization',
        '#16A085',
        [
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
                'Neural\nArchitecture Search',
                'https://paperswithcode.com/methods/category/neural-architecture-search',
                'https://www.youtube.com/watch?v=sROrvtXnT7Q',
            ),
        ],
    ),
    (
        'Tools (PyTorch)',
        '#5D6D7E',
        [
            (
                'PyTorch',
                'https://docs.pytorch.org/tutorials/',
                'https://www.youtube.com/watch?v=EMXfZB8FVUA',
            ),
            (
                'TensorBoard',
                'https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html',
                'https://www.youtube.com/watch?v=RLqsxWaQdHE',
            ),
            (
                'MLFlow',
                'https://mlflow.org/docs/latest/',
                'https://www.youtube.com/watch?v=3VFneBfMBJk',
            ),
            (
                'Hugging Face\nTransformers',
                'https://huggingface.co/docs/transformers/en/index',
                'https://www.youtube.com/watch?v=00GKzGyWFEs',
            ),
        ],
    ),
]

# Layout constants
SVG_WIDTH = 1100
NODE_W = 160
NODE_H = 72
NODE_RX = 10
COLS = 5
COL_GAP = 20
ROW_GAP = 16
SECTION_PADDING = 20
HEADER_H = 36
SECTION_GAP = 28
MARGIN_X = 30
MARGIN_TOP = 60

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', Arial, sans-serif"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
  r = int(hex_color[1:3], 16)
  g = int(hex_color[3:5], 16)
  b = int(hex_color[5:7], 16)
  return f'rgba({r},{g},{b},{alpha})'


def _section_height(n_concepts: int) -> int:
  rows = (n_concepts + COLS - 1) // COLS
  return HEADER_H + SECTION_PADDING + rows * NODE_H + (
      rows - 1) * ROW_GAP + SECTION_PADDING


def _render_node(
    x: int,
    y: int,
    concept: str,
    article_url: str,
    video_url: str,
    color: str,
) -> str:
  fill = _hex_to_rgba(color, 0.12)
  lines = concept.split('\n')
  # vertically center the concept name lines
  n_lines = len(lines)
  line_h = 15
  text_block_h = n_lines * line_h
  # link row sits 8px above bottom of node
  link_y = y + NODE_H - 14
  name_start_y = y + (NODE_H - text_block_h - 22) // 2 + line_h

  name_els = ''
  for i, line in enumerate(lines):
    name_els += (f'<tspan x="{x + NODE_W // 2}" dy="{0 if i == 0 else line_h}">'
                 f'{line}</tspan>')

  return textwrap.dedent(f'''\
    <g>
      <rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="{NODE_RX}"
            fill="{fill}" stroke="{color}" stroke-width="1.5"/>
      <text x="{x + NODE_W // 2}" y="{name_start_y}"
            font-family="{FONT_FAMILY}" font-size="12" font-weight="600"
            fill="#1a1a2e" text-anchor="middle" dominant-baseline="auto">
        {name_els}
      </text>
      <a href="{article_url}" target="_blank">
        <text x="{x + NODE_W // 2 - 28}" y="{link_y}"
              font-family="{FONT_FAMILY}" font-size="10" fill="{color}"
              text-anchor="middle" text-decoration="underline"
              style="cursor:pointer">article</text>
      </a>
      <text x="{x + NODE_W // 2 - 2}" y="{link_y}"
            font-family="{FONT_FAMILY}" font-size="10" fill="#888"
            text-anchor="middle">·</text>
      <a href="{video_url}" target="_blank">
        <text x="{x + NODE_W // 2 + 22}" y="{link_y}"
              font-family="{FONT_FAMILY}" font-size="10" fill="#c00"
              text-anchor="middle" text-decoration="underline"
              style="cursor:pointer">▶ video</text>
      </a>
    </g>''')


def _render_section(
    section_x: int,
    section_y: int,
    section_w: int,
    name: str,
    color: str,
    concepts: list,
) -> str:
  fill = _hex_to_rgba(color, 0.07)
  h = _section_height(len(concepts))
  parts = [
      f'<rect x="{section_x}" y="{section_y}" width="{section_w}" height="{h}" '
      f'rx="14" fill="{fill}" stroke="{color}" stroke-width="1.5" stroke-opacity="0.5"/>',
      f'<text x="{section_x + section_w // 2}" y="{section_y + 24}" '
      f'font-family="{FONT_FAMILY}" font-size="14" font-weight="700" '
      f'fill="{color}" text-anchor="middle" letter-spacing="0.5">{name}</text>',
  ]

  nodes_top = section_y + HEADER_H + SECTION_PADDING
  inner_w = section_w - 2 * SECTION_PADDING
  cols = min(COLS, len(concepts))
  total_nodes_w = cols * NODE_W + (cols - 1) * COL_GAP
  left_pad = section_x + SECTION_PADDING + max(0,
                                               (inner_w - total_nodes_w) // 2)

  for i, (concept, article_url, video_url) in enumerate(concepts):
    col = i % COLS
    row = i // COLS
    nx = left_pad + col * (NODE_W + COL_GAP)
    ny = nodes_top + row * (NODE_H + ROW_GAP)
    parts.append(_render_node(nx, ny, concept, article_url, video_url, color))

  return '\n'.join(parts)


def generate() -> str:
  section_w = SVG_WIDTH - 2 * MARGIN_X

  total_h = MARGIN_TOP
  for _, _, concepts in ROADMAP:
    total_h += _section_height(len(concepts)) + SECTION_GAP
  total_h += 40  # bottom padding

  parts = [
      f'<svg xmlns="http://www.w3.org/2000/svg" '
      f'width="{SVG_WIDTH}" height="{total_h}" '
      f'viewBox="0 0 {SVG_WIDTH} {total_h}">',
      '<defs>'
      '<style>'
      'a text { cursor: pointer; }'
      'a text:hover { opacity: 0.75; }'
      '</style>'
      '</defs>',
      f'<rect width="{SVG_WIDTH}" height="{total_h}" fill="#fafafa"/>',
      f'<text x="{SVG_WIDTH // 2}" y="40" '
      f'font-family="{FONT_FAMILY}" font-size="22" font-weight="800" '
      f'fill="#1a1a2e" text-anchor="middle" letter-spacing="1">Deep Learning Roadmap</text>',
  ]

  y = MARGIN_TOP
  for cat_name, color, concepts in ROADMAP:
    parts.append(
        _render_section(MARGIN_X, y, section_w, cat_name, color, concepts))
    y += _section_height(len(concepts)) + SECTION_GAP

  parts.append('</svg>')
  return '\n'.join(parts)


if __name__ == '__main__':
  svg = generate()
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT_PATH.write_text(svg, encoding='utf-8')
  print(f'Written to {OUTPUT_PATH}')
