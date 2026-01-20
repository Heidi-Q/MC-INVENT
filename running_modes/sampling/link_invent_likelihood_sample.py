from typing import List

import torch.utils.data as tud
from reinvent_models.lib_invent.models.dataset import DecoratorDataset
from reinvent_models.link_invent.dataset.paired_dataset import PairedDataset
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.reinforcement_learning.actions import BaseAction
# import abc
from typing import List

import numpy as np
import torch.utils.data as tud
from reinvent_chemistry import Conversions, TransformationTokens
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.link_invent.dataset.dataset import Dataset

from running_modes.reinforcement_learning.actions import BaseAction
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.sampling.logging.local_sampling_logger import LocalSamplingLogger

class LinkInventLikelihoodSampleModel:
    def __init__(self, model: GenerativeModelBase, batch_size: int, logger=None, randomize=False, sample_uniquely=True, with_likelihood=True):
        """
        Creates an instance of SampleModel.
        :params model: A model instance.
        :params batch_size: Batch size to use.
        :return:
        """
        self.logger: LocalSamplingLogger = logger
        self.model = model
        self._batch_size = batch_size
        self._bond_maker = BondMaker()
        self._randomize = randomize
        self._sample_uniquely = sample_uniquely
        self._with_likelihood = with_likelihood

        self._conversions = Conversions()
        self._attachment_points = AttachmentPoints()
        self._tokens = TransformationTokens()

    def run(self, warheads_list: List[str]) -> List[SampledSequencesDTO]:
        """
        Samples the model for the given number of SMILES.
        :params warheads_list: A list of warhead pair SMILES.
        :return: A list of SampledSequencesDTO.
        """
        warheads_list = self._randomize_warheads(warheads_list) if self._randomize else warheads_list
        clean_warheads = [self._attachment_points.remove_attachment_point_numbers(warheads) for warheads in warheads_list]
        dataset = Dataset(clean_warheads, self.model.get_vocabulary().input)
        data_loader = tud.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=dataset.collate_fn)

        for batch in data_loader:
            sampled_sequences = []
            for _ in range(self._batch_size):
                sampled_sequences.extend(self.model.sample(*batch))

            if self._sample_uniquely:
                sampled_sequences = self._sample_unique_sequences(sampled_sequences)
            # if self._with_likelihood:
            #     # warheads_batch, linker_batch, actor_nlls = self._calculate_likelihood(sampled_sequences)
            #     return self._calculate_likelihood(sampled_sequences)
            # else:
            return sampled_sequences

    def _sample_unique_sequences(self, sampled_sequences: List[SampledSequencesDTO]) -> List[SampledSequencesDTO]:
        # TODO could be part of a base sample action as it is the same for link and lib invent
        strings = ["".join([ss.input, ss.output]) for index, ss in enumerate(sampled_sequences)]
        unique_idxs = get_indices_of_unique_smiles(strings)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()

    def _randomize_warheads(self, warhead_pair_list: List[str]):
        randomized_warhead_pair_list = []
        for warhead_pair in warhead_pair_list:
            warhead_list = warhead_pair.split(self._tokens.ATTACHMENT_SEPARATOR_TOKEN)
            warhead_mol_list = [self._conversions.smile_to_mol(warhead) for warhead in warhead_list]
            warhead_randomized_list = [self._conversions.mol_to_random_smiles(mol) for mol in warhead_mol_list]
            # Note do not use self.self._bond_maker.randomize_scaffold, as it would add unwanted brackets to the
            # attachment points (which are not part of the warhead vocabulary)
            warhead_pair_randomized = self._tokens.ATTACHMENT_SEPARATOR_TOKEN.join(warhead_randomized_list)
            randomized_warhead_pair_list.append(warhead_pair_randomized)
        return randomized_warhead_pair_list

    def _calculate_likelihood(self, sampled_sequence_list: List[SampledSequencesDTO]):
        """
        Calculates the NLL for a set of SMILES strings.
        :param sampled_sequence_list: List wof sampled sequences (dto).
        :return: A tuple that follows the same order as the input list of SampledSequencesDTO.
        """
        input_output_list = [[ss.input, ss.output] for ss in sampled_sequence_list]
        dataset = PairedDataset(input_output_list, self.model.get_vocabulary())
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn,
                                    shuffle=False)
        for input_batch, output_batch in dataloader:
            nll = self.model.likelihood(*input_batch, *output_batch)
            return input_batch, output_batch, nll
            # return sampled_sequence_list, nll