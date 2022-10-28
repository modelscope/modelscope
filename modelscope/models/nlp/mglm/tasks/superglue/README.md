# Use GLM for your NLU tasks
To use GLM for your own NLU tasks, you should implement a subclass of `DataProcessor` in [tasks/superglue/dataset.py](dataset.py) and a subclass of `PVP` in [tasks/superglue/pvp.py](pvp.py). You should also specify the  We will take the RTE and ReCoRD tasks in SuperGLUE as an example.

## 1. Design your patterns
RTE is an NLI task in which the model is required to predict text entailment between a premise and a hypothesis. The label can be `entailment` or `not_entailment` One sample from the training set is
```
premise: No Weapons of Mass Destruction Found in Iraq Yet.
hypothesis: Weapons of Mass Destruction Found in Iraq.
label: not_entailment
```
We design the pattern as
```
"`hypothesis`"?, [MASK], "`premise`"
```
GLM predicts "Yes" for `entailment` and "No" for `not_entailment`. "Yes" and "No" are called verbalizers for `entailment` and `not_entailment`.

ReCoRD is a multi-choice QA task. Each example consists of a news article and a Cloze-style question about the article in which one entity is masked out. The system must predict the masked out entity from a list of possible entities in the provided passage. We directly adopt the cloze-style question as our pattern and use GLM to predict the masked entity.

## 2. Implement subclass of `DataProcessor`
A subclass of `DataProcessor` should implement `get_train_examples`, `get_dev_examples` and `get_test_examples`, which return the examples of the train, dev, and test sets. The returned value is a list of `InputExample`. It should also implement `get_labels` to return the list of possible labels. Hete we take the `RTEProcessor` as an example:
```python
class RteProcessor(DataProcessor):
    """Processor for the RTE data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str, hypothesis_name: str = "hypothesis",
                         premise_name: str = "premise") -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = line_idx
                label = example_json.get('label')
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples
```
After that, you should add the implemented class to ``PROCESSORS`` at the end of [tasks/superglue/dataset.py](dataset.py):
```python
PROCESSORS = {
    ...
    "rte": RteProcessor
}
```

## 3. Implement subclass of `PVP`
To implement a subclass of `PVP`, you should first decide your verbalizers is single-token or multi-token. The verbalizers in RTE, "Yes" and "No" are single-token. Instead, the verbalizers in ReCoRD are multi-token, as one entity can be tokenized into multiple tokens with WordPiece or BPE tokenizer.

For single-token task, you should set `is_multi_token=False` in the class definition. You should implement `get_parts` to return the inputs to GLM given an example and `verbalize` to return the verbalizer given a label. Take `RTEPVP` as an example:
```python
class RtePVP(PVP):
    is_multi_token = False
    VERBALIZER = {
        "not_entailment": [" No"],
        "entailment": [" Yes"]
    }

    @property
    def spell_length(self):
        return self.pattern_id

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = example.text_a
        text_b = example.text_b.rstrip(string.punctuation)
        return ['"', self.shortenable(text_b), '" ?'], [[self.mask], ', "', self.shortenable(text_a), '"']

    def verbalize(self, label) -> List[str]:
        return RtePVP.VERBALIZER[label]
```
We use `PvP.shortenable` to mark the segments that can be truncated when exceeding the maximum sequence length.

For multi-token task, you should set `is_multi_token=True` in the class definition. You should implement `get_parts` to return the inputs to GLM given an example and `get_answers` to return the candidates. Take `ReCoRDPVP` as an example:
```python
class RecordPVP(PVP):
    is_multi_token = True

    def get_answers(self, example: InputExample):
        choices = example.meta['candidates']
        choices = [" " + choice for choice in choices]
        return choices

    def get_parts(self, example: InputExample) -> FilledPattern:
        premise = self.shortenable(example.text_a)

        assert '@placeholder' in example.text_b, f'question "{example.text_b}" does not contain a @placeholder token'
        question_a, question_b = example.text_b.split('@placeholder')
        return [premise, " " + question_a.rstrip(), [self.mask], question_b], []
```
After that, you should implement the class to `PVPS` at the end of [tasks/superglue/pvp.py](pvp.py):
```python
PVPS = {
    ...
    'rte': RtePVP,
    'record': RecordPVP
}
```
## 4. Run the experiment
To run the experiment for your new task, you should create a config file like [config_tasks/task_rte.sh](/config_tasks/task_rte.sh). You should also specify the evaluation metrics for the task in `DEFAULT_METRICS` of [tasks/superglue/finetune.py](finetune.py):
```python
DEFAULT_METRICS = {
    ...
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "rte": [("accuracy", accuracy_metric)]
}
```
Then you can run the experiment with [finetune_superglue.sh](/scripts/finetune_superglue.sh):
```shell
bash scripts/finetune_superglue.sh \
     config_tasks/model_blocklm_large.sh \
     config_tasks/task_rte.sh
```
