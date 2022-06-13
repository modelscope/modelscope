from modelscope.models import SbertForNLI
from modelscope.pipelines import pipeline
from modelscope.preprocessors import NLIPreprocessor

model = SbertForNLI('../nlp_structbert_nli_chinese-base')
print(model)
tokenizer = NLIPreprocessor(model.model_dir)

semantic_cls = pipeline('nli', model=model, preprocessor=tokenizer)
print(type(semantic_cls))

print(
    semantic_cls(
        input=('我想还有一件事也伤害到了老师的招聘，那就是他们在课堂上失去了很多的权威',
               '教师在课堂上失去权威，导致想要进入这一职业的人减少了。')))
