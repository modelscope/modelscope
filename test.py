from modelscope.models import SbertForNLI
from modelscope.pipelines import pipeline
from modelscope.preprocessors import NLIPreprocessor

model = SbertForNLI('../nlp_structbert_nli_chinese-base')
print(model)
tokenizer = NLIPreprocessor(model.model_dir)

semantic_cls = pipeline('nli', model=model, preprocessor=tokenizer)
print(type(semantic_cls))

print(semantic_cls(input=('相反，这表明克林顿的敌人是疯子。', '四川商务职业学院商务管理在哪个校区？')))
