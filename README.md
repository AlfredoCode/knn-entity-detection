# Named Entity Recognition in Historical Czech Texts

This repository contains materials for a project focused on **Named Entity Recognition (NER)** in **historical Czech documents**, particularly digitized newspapers and chronicles.

## Data

The experiments are based on several datasets:

- **CNEC (Czech Named Entity Corpus)** – reference dataset for Czech NER.
- **Historical newspaper corpus** – OCR-processed historical Czech texts.
- **SYN v4** – large corpus of written Czech used for supplementary language modeling.

## Dataset generator
First unzip both datasets inside resources/
After that run following command:
`python3 -m src.data.dataset_generator`

## Fine tune RobeCzech + test
To fine tune the model, run following command:
`python3 -m src.models.finetuned_robeczech_cnec`

After that you can run `python3 -m src.models.finetuned_robeczech_cnec_test` to see evaluation results.

## Inter Anotator Agreement
Run `python3 get_iaa.py` inside experiments folder.


## Authors

Bc. Dominik Hofman

Bc. Lucie Jadrná

Bc. Vítězslav Cupl

Supervisor: Ing. Marek Vaško
