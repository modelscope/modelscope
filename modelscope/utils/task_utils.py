from modelscope.metainfo import TaskModels
from modelscope.utils import registry
from modelscope.utils.constant import Tasks

SUB_TASKS = 'sub_tasks'
PARENT_TASK = 'parent_task'
TASK_MODEL = 'task_model'

DEFAULT_TASKS_LEVEL = {
    Tasks.text_classification: {
        SUB_TASKS: [
            Tasks.text_classification,
            Tasks.sentence_similarity,
            Tasks.sentiment_classification,
            Tasks.sentiment_analysis,
            Tasks.nli,
        ],
        TASK_MODEL:
        TaskModels.text_classification,
    },
    Tasks.token_classification: {
        SUB_TASKS: [
            Tasks.token_classification,
            Tasks.named_entity_recognition,
            Tasks.word_segmentation,
            Tasks.part_of_speech,
        ],
        TASK_MODEL:
        TaskModels.text_classification,
    },
    Tasks.token_classification: {
        SUB_TASKS: [
            Tasks.token_classification,
            Tasks.named_entity_recognition,
            Tasks.word_segmentation,
            Tasks.part_of_speech,
        ],
        TASK_MODEL:
        TaskModels.text_classification,
    },
    Tasks.text_generation: {
        SUB_TASKS: [
            Tasks.text_generation,
            Tasks.text2text_generation,
        ],
        TASK_MODEL: TaskModels.text_generation,
    },
    Tasks.information_extraction: {
        SUB_TASKS: [
            Tasks.information_extraction,
            Tasks.relation_extraction,
        ],
        TASK_MODEL: TaskModels.information_extraction,
    },
    Tasks.fill_mask: {
        SUB_TASKS: [
            Tasks.fill_mask,
        ],
        TASK_MODEL: TaskModels.fill_mask,
    },
    Tasks.text_ranking: {
        SUB_TASKS: [
            Tasks.text_ranking,
        ],
        TASK_MODEL: TaskModels.text_ranking,
    }
    # TODO: add other tasks with their sub tasks in different domains
}


def _inverted_index(forward_index):
    inverted_index = dict()
    for index in forward_index:
        for item in forward_index[index][SUB_TASKS]:
            inverted_index[item] = {
                PARENT_TASK: index,
                TASK_MODEL: forward_index[index][TASK_MODEL],
            }
    return inverted_index


INVERTED_TASKS_LEVEL = _inverted_index(DEFAULT_TASKS_LEVEL)


def get_task_by_subtask_name(group_key):
    if group_key in INVERTED_TASKS_LEVEL:
        return INVERTED_TASKS_LEVEL[group_key][
            PARENT_TASK], INVERTED_TASKS_LEVEL[group_key][TASK_MODEL]
    else:
        return group_key, None
