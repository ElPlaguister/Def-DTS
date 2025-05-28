# Def-DTS

Official code for ACL 2025 paper: [Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](https://arxiv.org/abs/2505.21033).

## Overview

![Image](https://github.com/user-attachments/assets/9789d5f8-bd02-4dae-8c2b-453bdde5369e)

This is the overview of Def-DTS framework.

- We introduce LLM reasoning techniques to DTS for the first time, consolidating insights from previous methodologies into a coherent and deductive prompt design.

- We reformulate DTS as an utterance-level intent classification task by implementing intent classification as a core component of a multi-step reasoning process, enabling flexible and task-agnostic prompting.

- Our method empirically demonstrates superior performance across nearly all comparative baselines, underscoring the efficacy of prompt engineering in DTS.

- Through an in-depth analysis of our approach's reasoning results, we shed light on the challenges LLM reasoning faces in DTS and discuss the possibility of using LLM as a DTS auto-labeler.

Please refer to the paper for details.

## Citation

```bibtex
@misc{lee2025defdts,
  title={Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation},
  author={Seungmin Lee and Yongsang Yoo and Minhwa Jung and Min Song},
  year={2025},
  eprint={2505.21033},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
