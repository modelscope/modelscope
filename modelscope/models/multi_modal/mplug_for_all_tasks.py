import os.path as osp
from typing import Dict, List

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['MPlugForAllTasks']


@MODELS.register_module(
    Tasks.visual_question_answering, module_name=Models.mplug)
@MODELS.register_module(Tasks.image_captioning, module_name=Models.mplug)
@MODELS.register_module(Tasks.image_text_retrieval, module_name=Models.mplug)
class MPlugForAllTasks(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the mplug model from the `model_dir` path.
        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from modelscope.models.multi_modal.mplug import MPlug
        self.model = MPlug.from_pretrained(model_dir)
        self.tokenizer = self.model.tokenizer

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]),
                    }
        """

        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''),
                               ('[unused1]', ''), (r' +', ' '), ('[SEP]', ''),
                               ('[unused2]', ''), ('[CLS]', ''), ('[UNK]', ''))

        # get task from config file
        task = Config.from_file(
            osp.join(self.model_dir, ModelFile.CONFIGURATION)).task

        # inference
        if not self.training and 'question' in input:
            output = self.model(input['image'], input['question'], train=False)
            if task == Tasks.image_text_retrieval:
                return {OutputKeys.SCORES: output[0].tolist()}
            topk_ids, _ = output
            topk_ids = [topk_ids[i][0] for i in range(len(topk_ids))]
            pred_strings: List[str] = \
                self.tokenizer.batch_decode(topk_ids, skip_special_tokens=True)
            output = []
            for pred_string in pred_strings:
                for _old, _new in replace_tokens_bert:
                    pred_string = pred_string.replace(_old, _new)
                pred_string = pred_string.strip()
                output.append(pred_string)
            output_key = OutputKeys.CAPTION \
                if task == Tasks.image_captioning else OutputKeys.TEXT
            return {output_key: output}

        # train and evaluate
        import addict
        image = input['image']
        answer = addict.Dict(
            input_ids=input['answer_input_ids'],
            attention_mask=input['answer_attention_mask'])
        if 'index' not in input:
            question = addict.Dict(
                input_ids=input['question_input_ids'],
                attention_mask=input['question_attention_mask'])
            output = self.model(image, question, answer, train=self.training)
        else:
            index = input['index']
            output = self.model(image, answer, index, train=self.training)
        if self.training:
            return {OutputKeys.LOSS: output}

        # evaluate
        topk_ids, _ = output
        preds: List[str] = [
            self.tokenizer.decode(batch[0]) for batch in topk_ids
        ]
        for i in range(len(preds)):
            for _old, _new in replace_tokens_bert:
                preds[i] = preds[i].replace(_old, _new)
            preds[i] = preds[i].strip()
        tgts: List[str] = [
            self.tokenizer.decode(batch)
            for batch in input['answer_input_ids'].cpu().numpy().tolist()
        ]
        for i in range(len(tgts)):
            for _old, _new in replace_tokens_bert:
                tgts[i] = tgts[i].replace(_old, _new)
            preds[i] = preds[i].strip()
        return {'preds': preds, 'tgts': tgts}
