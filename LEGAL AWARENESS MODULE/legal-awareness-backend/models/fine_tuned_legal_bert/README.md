---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:7810
- loss:MultipleNegativesRankingLoss
base_model: law-ai/InLegalBERT
widget:
- source_sentence: What was the impact of the COVID-19 pandemic on the contract between
    Municipal Council Gondia and Divi Works & Suppliers, HUF & Ors.?
  sentences:
  - The COVID-19 pandemic led to financial constraints and the closure of schools,
    prompting the Government of Maharashtra to advise against non-priority expenditure,
    which resulted in the suspension and eventual cancellation of the work order by
    the Municipal Council Gondia.
  - Section 25(3) of the Indian Contract Act allows a promise to pay a time-barred
    debt if it is in writing and signed. The appellant argued that the cheque issued
    was a valid promise under this section.
  - The High Court of Uttarakhand allowed the writ petition filed by the landlords,
    upholding the declaration of vacancy and the order of possession in favor of the
    landlords.
- source_sentence: What was the Supreme Court's view on whether the case fell under
    Section 302 or Section 304 of the IPC?
  sentences:
  - Devendra Kumar Saxena filed the Transfer Petitions on the grounds of his advanced
    age (about 70 years) and severe health conditions, including a paralytic attack
    and brain stroke, which made it difficult for him to move, walk, speak, and perform
    routine activities without assistance.
  - The Supreme Court noted that extra-judicial confessions are a weak form of evidence
    and cannot be relied upon solely for conviction without corroborative evidence.
    In this case, the extra-judicial confessions were not corroborated by any reliable
    evidence.
  - The Supreme Court held that the case did not fall under Section 302 IPC but under
    Section 304 Part II IPC, as there was no intention to kill, and the appellant's
    actions were provoked by a quarrel the previous day.
- source_sentence: What sections of the Indian Penal Code (IPC) were the accused charged
    under?
  sentences:
  - The Special Judge noted that the victim's statements and the case diary provided
    sufficient reasons to suspect that the sexual abuse occurred, emphasizing the
    relationship between the victim and the accused and the trauma faced by the victim.
  - The Supreme Court concluded that the evidence of PW-1, being the eyewitness, remained
    unimpeachable and credible. The Court did not find any reason to discard his testimony
    solely because he did not raise an alarm or intervene during the incident.
  - The accused were charged under Sections 147, 148, 323/149, 324/149, 326/149, 302/149,
    and 379 of the IPC.
- source_sentence: What were the grounds for the Supreme Court's decision in favor
    of Raj Kumar Bhatia regarding the amendment of the written statement?
  sentences:
  - The decision may set a precedent for fair market value assessment in future land
    acquisition cases, ensuring that compensation reflects the true market value based
    on comparable sales.
  - There were 298 prosecution witnesses listed in the charge sheet, but the respondent
    indicated that only 100 to 105 witnesses were likely to be examined.
  - The Supreme Court held that the proposed amendment was an elaboration of what
    was stated in the original written statement and that the High Court had overstepped
    its jurisdiction under Article 227 of the Constitution by evaluating the merits
    of the case while considering the amendment application.
- source_sentence: What was the argument of B. S. Hari's counsel regarding the charges?
  sentences:
  - The Supreme Court observed that there was no evidence to prove that Tarlochan
    Singh @ Rana willingly handed over the firearm to Gurpreet Singh @ Titu, who took
    it without his permission.
  - Section 156(3) of the Cr.P.C. allows a Magistrate to order an investigation by
    the police. The controversy in this case revolves around whether a Magistrate
    can order such an investigation without prior sanction when public servants are
    involved.
  - The counsel argued that the charges against B. S. Hari were not proved, as the
    main accused were either given relief or discharged due to lack of evidence. It
    was also argued that the trial was a nullity without proper sanction from the
    Central Government.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on law-ai/InLegalBERT

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) <!-- at revision b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "What was the argument of B. S. Hari's counsel regarding the charges?",
    'The counsel argued that the charges against B. S. Hari were not proved, as the main accused were either given relief or discharged due to lack of evidence. It was also argued that the trial was a nullity without proper sanction from the Central Government.',
    'The Supreme Court observed that there was no evidence to prove that Tarlochan Singh @ Rana willingly handed over the firearm to Gurpreet Singh @ Titu, who took it without his permission.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.8465, -0.0295],
#         [ 0.8465,  1.0000,  0.0103],
#         [-0.0295,  0.0103,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,810 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 23.65 tokens</li><li>max: 61 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 46.09 tokens</li><li>max: 159 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                   |
  |:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What procedural error did the defense highlight in the case of Parvat Singh & Ors. vs. State of Madhya Pradesh?</code> | <code>The defense highlighted that no torch was recovered from the scene despite the eyewitness claiming to have seen the accused in torchlight. They also pointed out that the High Court erroneously relied on the statement recorded under Section 161 Cr.P.C., which is inadmissible as evidence.</code> |
  | <code>How did the Supreme Court view the suspicious circumstances surrounding the Will?</code>                               | <code>The Supreme Court noted the unexplained suspicious circumstances and discrepancies in the Will, such as differences in the color of the paper, variations in the signatures, and the use of different pens for signing, and concluded that the Will was not genuine.</code>                            |
  | <code>How did the Supreme Court interpret the applicability of Section 42(4) to captive consumers?</code>                    | <code>The Supreme Court interpreted that Section 42(4) is applicable only when the State Commission permits consumers to receive electricity from sources other than their distribution licensee, which does not apply to captive consumers who have statutory rights under Section 9.</code>                |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.0225 | 500  | 0.3945        |
| 2.0450 | 1000 | 0.0593        |


### Framework Versions
- Python: 3.9.6
- Sentence Transformers: 5.1.2
- Transformers: 4.57.6
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->