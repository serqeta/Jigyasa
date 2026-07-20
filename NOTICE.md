# Third-Party Model Notices

VoiceShield loads the following pretrained checkpoints at runtime. Weights
are downloaded by `scripts/download_model.py` and are not redistributed in
this repository.

| Component | Model | Source | License |
|---|---|---|---|
| nii (primary) | MMS-300M anti-deepfake | huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake | **CC BY-NC-SA 4.0 — NON-COMMERCIAL.** Fine for hackathon/research; a commercial deployment must license or replace this component. Weights converted to transformers format by scripts/convert_nii.py (share-alike applies to the conversion). |
| stage1 (retired) | AASIST-L | github.com/clovaai/aasist | MIT |
| ssl | wav2vec2 XLS-R 300M deepfake classifier | huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | Apache-2.0 |
| spec | AST fine-tuned on ASVspoof 5 | huggingface.co/MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection | BSD-3-Clause |
| wavlm | WavLM-base fine-tuned on In-the-Wild | huggingface.co/abhishtagatya/wavlm-base-960h-itw-deepfake | No license declared on model card — verify before commercial deployment |

Notes:

- The `wavlm` component has no license stated by its author. It is used
  here for hackathon/research evaluation; obtain clarification or disable
  it (`config.HF_SCORERS["wavlm"]["enabled"] = False`) for production use.
- Base architectures carry their own upstream licenses: WavLM
  (microsoft/unilm, MIT), XLS-R (fairseq, MIT), AST (MIT).
