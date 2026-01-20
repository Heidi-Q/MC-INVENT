from typing import List
import os
import numpy as np
import tqdm
### Raw code
# import reinvent_models.reinvent_core.models.model as reinvent
### my code
# import reinvent_models.link_invent.link_invent_model as lkim
# import reinvent_models.model_factory.link_invent_adapter as lia
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
# from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
# from running_modes.reinforcement_learning.actions import LinkInventLikelihoodEvaluation, LinkInventSampleModel
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.constructors.base_running_mode import BaseRunningMode
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry import Conversions
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.compound_sampling.link_invent_sample_from_model_configuration import LinkInventSampleFromModelConfiguration
from running_modes.sampling.logging.sampling_logger import SamplingLogger
from running_modes.sampling.link_invent_likelihood_sample import LinkInventLikelihoodSampleModel

# class LinkInventSampleFromModelRunner(BaseRunningMode):
#     def
class LinkInventSampleFromModelRunner(BaseRunningMode):

    def __init__(self, main_config: GeneralConfigurationEnvelope, configuration: LinkInventSampleFromModelConfiguration):
        model_config = ModelConfiguration("link_invent", "inference", configuration.model_path)
        self.model = GenerativeModel(model_config)
        self._output = self._open_output(path=configuration.output_smiles_path)
        # self._num_sampling = configuration.num_sampling
        self._sampling_num = configuration.sampling_num
        self._with_likelihood = configuration.with_likelihood
        self.warheads = self._read_input(input_smi=configuration.warheads)
        self._num_smiles = len(self.warheads)#*
        self._batch_size = configuration.batch_size
        self._logger = SamplingLogger(main_config)
        self._attachment_points = AttachmentPoints()
        self._bond_maker = BondMaker()
        self._conversion = Conversions()
    def _open_output(self, path):
        try:
            os.mkdir(os.path.dirname(path))
        except FileExistsError:
            pass
        return open(path, "wt+")

    def _read_input(self, input_smi):
        if not os.path.exists(input_smi):
            return [input_smi]
        # if Chem.MolFromSmiles()
        try:
            warheads = []
            with open(input_smi, 'r') as f:
                for line in f.readlines():
                    warheads.append(line.strip())
        except FileExistsError:
            pass
        return warheads

    # def _recovery_target(self, path):
    #     try:
    #         macrocycles = []
    #         with open(path, 'r') as f:
    #             for line in f.readlines():
    #                 macrocycles.append(line.strip())
    #     except FileNotFoundError:
    #         pass
    #     return macrocycles

    def run(self):
        # for warhead in self._warheads:
        molecules_left = self._num_smiles
        self._output.write("macrocycles,warheads,linkers,nll\n")
        with tqdm.tqdm(total=self._num_smiles) as progress_bar:
            while molecules_left > 0:
                current_batch_size = min(self._batch_size, molecules_left)
                current_batch_smiles = self.warheads[:current_batch_size]
                sampling_action = LinkInventLikelihoodSampleModel(self.model, self._sampling_num, self._logger, current_batch_smiles, sample_uniquely=False)#self._warheads
                sampled_sequences = sampling_action.run(current_batch_smiles)#self._warheads)
                macrocycles = self._join_linker_and_warheads_my(sampled_sequences, keep_labels=False)
                # _,_,nlls = self._calculate_likelihood(sampled_sequences)
                # macrocycles = []
                # for molecule in molecules:
                #     smiles_str = self._conversion.mol_to_smiles(molecule) if molecule else "INVALID"
                #     smiles_str = self._conversion.mol_to_smiles(self._attachment_points.remove_attachment_point_numbers_from_mol(self._conversion.smile_to_mol(smiles_str))) if molecule else "INVALID"
                #     macrocycles.append(smiles_str)
                for macro, ss in zip(macrocycles, sampled_sequences):
                    output_row = [macro]
                    samp = [ss.input, ss.output, str(ss.nll)]
                    output_row.extend(samp)
                    self._output.write("{}\n".format(",".join(output_row)))
                molecules_left -= current_batch_size # self._batch_size
                self.warheads = self.warheads[current_batch_size-1:]
                progress_bar.update(current_batch_size)#self._batch_size
        self._output.close()
        self._logger.log_out_input_configuration()

    def _join_linker_and_warheads_my(self, sampled_sequences: List[SampledSequencesDTO], keep_labels=False):
        molecules = []
        for sample in sampled_sequences:
            linker = self._attachment_points.add_attachment_point_numbers(sample.output, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations_my(linker, sample.input, keep_labels_on_atoms=keep_labels)

            molecule = self._attachment_points.remove_attachment_point_numbers_from_mol(molecule) if molecule else None
            smiles_str = self._conversion.mol_to_smiles(molecule) if molecule else "INVALID"
            # smiles_str = self._conversion.mol_to_smiles(
            #     self._attachment_points.remove_attachment_point_numbers_from_mol(
            #         self._conversion.smile_to_mol(smiles_str))) if molecule else "INVALID"

            molecules.append(smiles_str)#molecule)
        return molecules

    # def _calculate_likelihood(self, sampled_sequences: List[SampledSequencesDTO]):
    #     nll_calculation_action = LinkInventLikelihoodEvaluation(self.model, self._logger)
    #     encoded_warheads, encoded_linkers, nlls = nll_calculation_action.run(sampled_sequences)
    #     return encoded_warheads, encoded_linkers, nlls
