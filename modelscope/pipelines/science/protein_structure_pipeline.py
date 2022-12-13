# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from typing import Any, Dict, List, Optional, Union

import json
import numpy as np
import torch
from unicore.utils import tensor_tree_map

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.science.unifold.config import model_config
from modelscope.models.science.unifold.data import protein, residue_constants
from modelscope.models.science.unifold.dataset import (UnifoldDataset,
                                                       load_and_process)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor, build_preprocessor
from modelscope.utils.constant import Fields, Frameworks, Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.hub import read_config
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ProteinStructurePipeline']


def automatic_chunk_size(seq_len):
    if seq_len < 512:
        chunk_size = 256
    elif seq_len < 1024:
        chunk_size = 128
    elif seq_len < 2048:
        chunk_size = 32
    elif seq_len < 3072:
        chunk_size = 16
    else:
        chunk_size = 1
    return chunk_size


def load_feature_for_one_target(
    config,
    data_folder,
    seed=0,
    is_multimer=False,
    use_uniprot=False,
    symmetry_group=None,
):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ['A']
        if use_uniprot:
            uniprot_msa_dir = data_folder

    else:
        uniprot_msa_dir = data_folder
        sequence_ids = open(
            os.path.join(data_folder, 'chains.txt'),
            encoding='utf-8').readline().split()

    if symmetry_group is None:
        batch, _ = load_and_process(
            config=config.data,
            mode='predict',
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            sequence_ids=sequence_ids,
            monomer_feature_dir=data_folder,
            uniprot_msa_dir=uniprot_msa_dir,
        )
    else:
        # Not for unifold-symmetry
        # only for unifold-multimer
        batch, _ = load_and_process(
            config=config.data,
            mode='predict',
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            sequence_ids=sequence_ids,
            monomer_feature_dir=data_folder,
            uniprot_msa_dir=uniprot_msa_dir,
        )
    batch = UnifoldDataset.collater([batch])
    return batch


@PIPELINES.register_module(
    Tasks.protein_structure, module_name=Pipelines.protein_structure)
class ProteinStructurePipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a protein structure pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the protein structure task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='protein-structure',
            >>>    model='DPTech/uni-fold-monomer')
            >>> protein = 'LILNLRGGAFVSNTQITMADKQKKFINEIQEGDLVRSYSITDETFQQNAVTSIVKHEADQLCQINFGKQHVVC'
            >>> print(pipeline_ins(protein))

        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.cfg = read_config(self.model.model_dir)
        self.config = model_config(
            self.cfg['pipeline']['model_name'])  # alphafold config
        self.postprocessor = self.cfg.pop('postprocessor', None)
        if preprocessor is None:
            preprocessor_cfg = self.cfg.preprocessor
            self.preprocessor = build_preprocessor(preprocessor_cfg,
                                                   Fields.science)
        self.model.eval()

    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def _process_single(self, input, *args, **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params', {})
        forward_params = kwargs.get('forward_params', {})
        postprocess_params = kwargs.get('postprocess_params', {})
        out = self.preprocess(input, **preprocess_params)
        with device_placement(self.framework, self.device_name):
            with torch.no_grad():
                out = self.forward(out, **forward_params)

        out = self.postprocess(out, **postprocess_params)
        return out

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        plddts = {}
        ptms = {}

        output_dir = os.path.join(self.preprocessor.output_dir_base,
                                  inputs['target_id'])

        pdbs = []
        for seed in range(self.cfg['pipeline']['times']):
            cur_seed = hash((42, seed)) % 100000
            batch = load_feature_for_one_target(
                self.config,
                output_dir,
                cur_seed,
                is_multimer=inputs['is_multimer'],
                use_uniprot=inputs['is_multimer'],
                symmetry_group=self.preprocessor.symmetry_group,
            )
            seq_len = batch['aatype'].shape[-1]
            self.model.model.globals.chunk_size = automatic_chunk_size(seq_len)

            with torch.no_grad():
                batch = {
                    k: torch.as_tensor(v, device='cuda:0')
                    for k, v in batch.items()
                }
                out = self.model(batch)

            def to_float(x):
                if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                    return x.float()
                else:
                    return x

            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
            batch = tensor_tree_map(to_float, batch)
            out = tensor_tree_map(lambda t: t[0, ...], out[0])
            out = tensor_tree_map(to_float, out)
            batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            plddt = out['plddt']
            mean_plddt = np.mean(plddt)
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1)
            # TODO: , may need to reorder chains, based on entity_ids
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors)
            cur_save_name = (f'{cur_seed}')
            plddts[cur_save_name] = str(mean_plddt)
            if inputs[
                    'is_multimer'] and self.preprocessor.symmetry_group is None:
                ptms[cur_save_name] = str(np.mean(out['iptm+ptm']))
            with open(os.path.join(output_dir, cur_save_name + '.pdb'),
                      'w') as f:
                f.write(protein.to_pdb(cur_protein))
                pdbs.append(protein.to_pdb(cur_protein))

        logger.info('plddts:' + str(plddts))
        model_name = self.cfg['pipeline']['model_name']
        score_name = f'{model_name}'
        plddt_fname = score_name + '_plddt.json'

        with open(os.path.join(output_dir, plddt_fname), 'w') as f:
            json.dump(plddts, f, indent=4)
        if ptms:
            logger.info('ptms' + str(ptms))
            ptm_fname = score_name + '_ptm.json'
            with open(os.path.join(output_dir, ptm_fname), 'w') as f:
                json.dump(ptms, f, indent=4)

        return pdbs

    def postprocess(self, inputs: Dict[str, Tensor], **postprocess_params):
        return inputs
