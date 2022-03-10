
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


class XconfigDimRangeComponentInit(XconfigLayerBase):
    """This class is for parsing lines like
     'dim-range-component name=feature1 input=Append(-3,0,3) dim=40 dim-offset=0'
    which will produce just a single component, of part of the input.
    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      dim=-1                   [Dimension of the output.]
      dim-offset=0             [Dimension offset of the input.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'dim-offset': 0 }

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']
        if self.config['dim'] <= 0:
            raise RuntimeError("'dim' must be specified and > 0.")
        elif self.config['dim'] > input_dim:
            raise RuntimeError("'dim' must be specified and lower than the input dim.")
        if self.config['dim-offset'] < 0 :
            raise RuntimeError("'dim-offset' must be specified and >= 0.")
        elif self.config['dim-offset'] + self.config['dim'] > input_dim:
            raise RuntimeError("'dim-offset' plus output dim must be lower than the input dim.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_dim = self.config['dim']
        if output_dim <= 0:
            self.config['dim'] = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['init', 'ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    def _generate_config(self):
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_node = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        dim_offset = self.config['dim-offset']

        configs = []
        line = ('dim-range-node name={0} input-node={1} dim={2} dim-offset={3}'.format(
            self.name, input_node, output_dim, dim_offset))
        configs.append(line)
        return configs


class XconfigElementwiseProductComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'elementwise-product-component name=noop1 input=Append(-3,0,3) dim=40'
    which will produce just a single component, of type NoOpComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
					   'dim': -1 }

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']
        if self.config['dim'] <= 0:
            raise RuntimeError("'dim' must be specified and > 0.")
        elif self.config['dim'] > input_dim:
            raise RuntimeError("'dim' must be specified and lower than the input dim.")
        elif input_dim % self.config['dim'] != 0:
            raise RuntimeError("input dim must be some times of the 'dim'.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_dim = self.config['dim']
        if output_dim <= 0:
            self.config['dim'] = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    def _generate_config(self):
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
		
        configs = []
        line = ('component name={0} type=ElementwiseProductComponent output-dim={1} input-dim={2}'.format(
            self.name, output_dim, input_dim))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs

class XconfigSimpleComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'simple-component name=scale type=FixedScaleComponent dim=1536 opts="scale=2 dim=1536"'
    which will produce just a single component, of type NoOpComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
					   'dim': -1,
					   'type': 'error',
					   'opts': 'error' }

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_dim = self.config['dim']
        if output_dim <= 0:
            self.config['dim'] = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    def _generate_config(self):
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        component_type = self.config['type']
        component_opts = self.config['opts']
				
        configs = []
        line = ('component name={0} type={1} {2}'.format(
            self.name, component_type, component_opts))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigCompositeComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'simple-component name=scale type=FixedScaleComponent dim=1536 opts="scale=2 dim=1536"'
    which will produce just a single component, of type NoOpComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
					   'dim': -1,
					   'max_rows_process': 2048,
					   'num_components': 1,
					   'component1': 'error',
					   'component2': 'error',
					   'component3': 'error',
					   'component4': 'error',
					   'component5': 'error',
					   'component6': 'error',
					   'component7': 'error',
					   'component8': 'error',
					   'component9': 'error',
					   'component10': 'error',
					   'component11': 'error'}

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_dim = self.config['dim']
        if output_dim <= 0:
            self.config['dim'] = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    def _generate_config(self):
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        max_rows_process = self.config['max_rows_process']
        num_components = self.config['num_components']
        
        sub_components = []
        for i in range(num_components):
            temp1 = 'component'+str(i)
            temp = self.config['temp1']
            temp = 'component'+str(i)+'='+"\'"+temp+"\'"
            sub_components.append(temp)

        sub_component_all = ' '.join(sub_components)
				
        configs = []
        line = ('component name={0} max_rows_process={1} num_components={2} {3}'.format(
            self.name, max_rows_process, num_components, sub_component_all))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs
