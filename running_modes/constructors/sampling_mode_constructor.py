from dacite import from_dict
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, SampleFromModelConfiguration, LinkInventSampleFromModelConfiguration
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.sampling.link_invent_sample_from_model import LinkInventSampleFromModelRunner
from running_modes.utils.general import set_default_device_cuda

# ### my code
class SamplingModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        set_default_device_cuda()
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()


        set_default_device_cuda()
        # runner = SampleFromModelRunner(self._configuration, config)

        if self._configuration.model_type == model_type.DEFAULT:
            config = from_dict(data_class=SampleFromModelConfiguration, data=self._configuration.parameters)
            runner = SampleFromModelRunner(self._configuration, config)

        # elif cls._configuration.model_type == model_type.LIB_INVENT:
        #     raise NotImplementedError(f"Running mode not implemented for a model type: {cls._configuration.model_type}")

        elif self._configuration.model_type == model_type.LINK_INVENT:
            ### raw code
            # config = from_dict(data_class=SampleFromModelConfiguration, data=self._configuration.parameters)
            ### my code
            config = from_dict(data_class=LinkInventSampleFromModelConfiguration, data=self._configuration.parameters)
            runner = LinkInventSampleFromModelRunner(self._configuration, config)

        # else:
        #     raise ValueError(f"Invalid model_type provided: '{cls._configuration.model_type}")

        return runner

# ### raw code
# class SamplingModeConstructor:
#     def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
#         self._configuration = configuration
#         config = from_dict(data_class=SampleFromModelConfiguration, data=self._configuration.parameters)
#         set_default_device_cuda()
#         runner = SampleFromModelRunner(self._configuration, config)
#         return runner