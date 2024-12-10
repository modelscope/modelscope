from pydantic import BaseModel


class ModelScopeRequest(BaseModel):

    def __init__(self, input: object, parameters: object):
        self.input = input
        self.parameters = parameters
