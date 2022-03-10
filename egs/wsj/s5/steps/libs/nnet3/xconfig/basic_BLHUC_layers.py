""" This module contains layers that just map to a single component.
"""

from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

class XconfigReluBLHUCBatchnormDropoutLayer(XconfigLayerBase):
    """This class is for parsing lines like
     'relu-renorm-layer name=layer1 dim=1024 input=Append(-3,0,3)'
    or:
     'sigmoid-layer name=layer1 dim=1024 input=Append(-3,0,3)'
    which specify addition of an affine component and a sequence of non-linearities.
    Here, the name of the layer itself dictates the sequence of nonlinearities
    that are applied after the affine component; the name should contain some
    combination of 'relu', 'renorm', 'sigmoid' and 'tanh',
    and these nonlinearities will be added along with the affine component.

    The dimension specified is the output dim; the input dim is worked out from the input descriptor.
    This class supports only nonlinearity types that do not change the dimension; we can create
    another layer type to enable the use p-norm and similar dimension-reducing nonlinearities.

    See other configuration values below.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      dim=-1                   [Output dimension of layer, e.g. 1024]
      bottleneck-dim=-1        [If you set this, a linear bottleneck is added, so
                                we project to first bottleneck-dim then to dim.  The
                                first of the two matrices is constrained to be
                                orthonormal.]
      speaker-id='[-1]'     [Input feature name of the speaker ID]
      speaker-count='[-1]'     [Input feature name of the speaker count]
      col-num=-1
      ng-blhuc-mean-options=''
      ng-blhuc-std-options=''
      ng-prior-mean-options=''
      ng-prior-std-options=''
      tied-std-scale=1.0
      KL-scale=0.0001
      self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
      learning-rate-factor=1.0   [This can be used to make the affine component
                                  train faster or slower].
      add-log-stddev=False     [If true, the log of the stddev of the output of
                                renorm layer is appended as an
                                additional dimension of the layer's output]
      l2-regularize=0.0       [Set this to a nonzero value (e.g. 1.0e-05) to
                               add l2 regularization on the parameter norm for
                                this component.
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'bottleneck-dim': -1,  # Deprecated!  Use tdnnf-layer for
                                              # factorized TDNNs, or prefinal-layer
                                              # for bottlenecks just before the output.
                       'self-repair-scale': 1.0e-05,
                       'speaker-id': '[-1]',
                       'speaker-count': '[-1]',
                       'col-num': -1,
                       'blhuc-activation': 'sigmoid', # 0: 2sigmoid; 1: linear; 2: exp
                       'ng-blhuc-mean-options': '',
                       'ng-blhuc-std-options': '',
                       'ng-prior-mean-options': '',
                       'ng-prior-std-options': '',
                       'tied-std-scale': 1.0,
                       'KL-scale': 0.0001,
                       'target-rms': 1.0,
                       'ng-affine-options': '',
                       'ng-linear-options': '',    # only affects bottleneck layers.
                       'dropout-proportion': 0.5,  # dropout-proportion only
                                                   # affects layers with
                                                   # 'dropout' in the name
                       'dropout-per-dim': False,  # if dropout-per-dim=true, the dropout
                                                  # mask is shared across time.
                       'dropout-per-dim-continuous':  False, # if you set this, it's
                                                    # like dropout-per-dim but with a
                                                    # continuous-valued (not zero-one) mask.
                       'add-log-stddev': False,
                       # the following are not really inspected by this level of
                       # code, just passed through to the affine component if
                       # their value is not ''.
                       'bias-stddev': '',
                       'l2-regularize': '',
                       'learning-rate-factor': '',
                       'max-change': 0.75 }

    def check_configs(self):
        if self.config['dim'] < 0:
            raise RuntimeError("dim has invalid value {0}".format(self.config['dim']))
        b = self.config['bottleneck-dim']
        if b >= 0 and (b >= self.config['dim'] or b == 0):
            raise RuntimeError("bottleneck-dim has an invalid value {0}".format(b))

        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if (self.config['learning-rate-factor'] != '' and
            self.config['learning-rate-factor'] <= 0.0):
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output is None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        # return something like: layer3.renorm
        return '{0}.{1}'.format(self.name, last_nonlinearity)

    def output_dim(self, auxiliary_output=None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
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
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, nonlinearities):
        output_dim = self.output_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        speaker_input = self.config['speaker-id']
        speaker_count = self.config['speaker-count']
        col_num = self.config['col-num']
        blhuc_activation = self.config['blhuc-activation']
        blhuc_mean_options = self.config['ng-blhuc-mean-options']
        blhuc_std_options = self.config['ng-blhuc-std-options']
        prior_mean_options = self.config['ng-prior-mean-options']
        prior_std_options = self.config['ng-prior-std-options']
        tied_std_scale = self.config['tied-std-scale']
        KL_scale = self.config['KL-scale']

        affine_options = self.config['ng-affine-options']
        for opt_name in [ 'max-change', 'learning-rate-factor',
                          'bias-stddev', 'l2-regularize' ]:
            value = self.config[opt_name]
            if value != '':
                affine_options += ' {0}={1}'.format(opt_name, value)

        # The output of the affine component needs to have one dimension fewer in order to
        # get the required output dim, if the final 'renorm' component has 'add-log-stddev' set
        # (since in that case it increases the dimension by one).
        if self.config['add-log-stddev']:
            output_dim -= 1
            if not self.layer_type.split('-')[-2] == "renorm":
                raise RuntimeError("add-log-stddev cannot be true unless "
                                   "there is a final 'renorm' component.")

        configs = []
        cur_dim = input_dim
        cur_node = input_desc

        # First the affine node (or linear then affine, if bottleneck).
        if self.config['bottleneck-dim'] > 0:
            # The 'bottleneck-dim' option is deprecated and may eventually be
            # removed.  Best to use tdnnf-layer if you want factorized TDNNs.

            # This is the bottleneck case (it doesn't necessarily imply we
            # will be using the features from the bottleneck; it's just a factorization
            # of the matrix into two pieces without a nonlinearity in between).
            # We don't include the l2-regularize option because it's useless
            # given the orthonormality constraint.
            linear_options = self.config['ng-linear-options']
            for opt_name in [ 'max-change', 'learning-rate-factor' ]:
                value = self.config[opt_name]
                if value != '':
                    linear_options += ' {0}={1}'.format(opt_name, value)

            bottleneck_dim = self.config['bottleneck-dim']
            # note: by default the LinearComponent uses natural gradient.
            line = ('component name={0}.linear type=LinearComponent '
                    'input-dim={1} orthonormal-constraint=1.0 output-dim={2} {3}'
                    ''.format(self.name, input_dim, bottleneck_dim, linear_options))
            configs.append(line)
            line = ('component-node name={0}.linear component={0}.linear input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.linear'.format(self.name)
            cur_dim = bottleneck_dim


        line = ('component name={0}.affine type=NaturalGradientAffineComponent'
                ' input-dim={1} output-dim={2} {3}'
                ''.format(self.name, cur_dim, output_dim, affine_options))
        configs.append(line)
        line = ('component-node name={0}.affine component={0}.affine input={1}'
                ''.format(self.name, cur_node))
        configs.append(line)
        cur_node = '{0}.affine'.format(self.name)

        for i, nonlinearity in enumerate(nonlinearities):
            if nonlinearity == 'relu':
                line = ('component name={0}.{1} type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'renorm':
                add_log_stddev = "false"
                if i == len(nonlinearities) - 1:
                    add_log_stddev = ("true" if self.config['add-log-stddev']
                                      else "false")
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ' add-log-stddev={4}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  target_rms, add_log_stddev))

            elif nonlinearity == 'batchnorm':
                line = ('component name={0}.{1}'
                        ' type=BatchNormComponent dim={2} target-rms={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  target_rms))

            elif nonlinearity == 'so':
                line = ('component name={0}.{1}'
                        ' type=ScaleAndOffsetComponent dim={2} max-change=0.5 '
                        ''.format(self.name, nonlinearity, output_dim))

            elif nonlinearity == 'dropout':
                if not (self.config['dropout-per-dim'] or
                        self.config['dropout-per-dim-continuous']):
                    line = ('component name={0}.{1} type=DropoutComponent '
                            'dim={2} dropout-proportion={3}'.format(
                                self.name, nonlinearity, output_dim,
                                self.config['dropout-proportion']))
                else:
                    continuous_opt='continuous=true' if self.config['dropout-per-dim-continuous'] else ''

                    line = ('component name={0}.dropout type=GeneralDropoutComponent '
                            'dim={1} dropout-proportion={2} {3}'.format(
                                self.name, output_dim, self.config['dropout-proportion'],
                                continuous_opt))
            elif nonlinearity == 'blhuc':
                line = ('component name={0}.BLHUC.mean type=LinearSelectColComponent '
                        'input-dim=1 output-dim={1} col-num={2} {3}'.format(self.name, output_dim, col_num, blhuc_mean_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.mean component={0}.BLHUC.mean input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.std1 type=LinearSelectColComponent '
                        'input-dim=1 output-dim=1 col-num={1} {2}'.format(self.name, col_num, blhuc_std_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std1 component={0}.BLHUC.std1 input={1}'.format(self.name, speaker_input))
                configs.append(line)
                line = ('component name={0}.BLHUC.std2 type=NoOpComponent '
                        'dim=1 backprop-scale={1}'.format(self.name, tied_std_scale))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std2 component={0}.BLHUC.std2 input={0}.BLHUC.std1'.format(self.name))
                configs.append(line)
                line = ('component name={0}.BLHUC.std type=CopyNComponent '
                        'input-dim=1 output-dim={1}'.format(self.name, output_dim))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std component={0}.BLHUC.std input={0}.BLHUC.std2'.format(self.name))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.prior_mean type=ConstantFunctionComponent '
                        'input-dim=1 output-dim={1} {2}'.format(self.name, output_dim, prior_mean_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.prior_mean component={0}.BLHUC.prior_mean input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.prior_std type=ConstantFunctionComponent '
                        'input-dim=1 output-dim={1} {2}'.format(self.name, output_dim, prior_std_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.prior_std component={0}.BLHUC.prior_std input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.vec type=BayesVecKLGaussianComponent '
                        'output-dim={1} input-dim={2} KL-scale={3} '
                        'input-frame-scale=true inv-frame-scale=true rand-per-frame=false KL-output=false test-mode=false'.format(self.name, output_dim, output_dim*4+1, KL_scale))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.vec component={0}.BLHUC.vec '
                        'input=Append({0}.BLHUC.mean, {0}.BLHUC.std, {0}.BLHUC.prior_mean, {0}.BLHUC.prior_std, {1})'.format(self.name, speaker_count))
                configs.append(line)
                
                if blhuc_activation == 'sigmoid':
                    line = ('component name={0}.BLHUC.sigmoid type=SigmoidComponent dim={1} self-repair-scale=0'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.sigmoid component={0}.BLHUC.sigmoid input={0}.BLHUC.vec'.format(self.name))
                    configs.append(line)
                elif blhuc_activation == 'linear':
                    line = ('component name={0}.BLHUC.noop type=NoOpComponent dim={1}'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.noop component={0}.BLHUC.noop input={0}.BLHUC.vec'.format(self.name))
                    configs.append(line)
                elif blhuc_activation == 'exp':
                    line = ('component name={0}.BLHUC.exp type=ExpComponent dim={1} self-repair-scale=0'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.exp component={0}.BLHUC.exp input={0}.BLHUC.vec'.format(self.name))
                    configs.append(line)
                line = ('component name={0}.product type=ElementwiseProductComponent output-dim={1} input-dim={2}'.format(self.name, output_dim, 2*output_dim))
                configs.append(line)
                if blhuc_activation == 'sigmoid':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, Scale(2.0, {0}.BLHUC.sigmoid))'.format(self.name, cur_node))
                elif blhuc_activation == 'linear':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, {0}.BLHUC.noop)'.format(self.name, cur_node))
                elif blhuc_activation == 'exp':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, {0}.BLHUC.exp)'.format(self.name, cur_node))
                configs.append(line)
                nonlinearity='product'
            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   .format(nonlinearity))

            if nonlinearity != 'product':
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))

                configs.append(line)
            cur_node = '{0}.{1}'.format(self.name, nonlinearity)
        return configs


class XconfigReluBLHUC1BatchnormDropoutLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'bottleneck-dim': -1,  # Deprecated!  Use tdnnf-layer for
                                              # factorized TDNNs, or prefinal-layer
                                              # for bottlenecks just before the output.
                       'self-repair-scale': 1.0e-05,
                       'speaker-id': '[-1]',
                       'speaker-count': '[-1]',
                       'col-num': -1,
                       'blhuc-activation': 'sigmoid', # 0: 2sigmoid; 1: linear; 2: exp
                       'ng-blhuc-mean-options': '',
                       'ng-blhuc-std-options': '',
                       'ng-prior-mean-options': '',
                       'ng-prior-std-options': '',
                       'tied-std-scale': 1.0,
                       'KL-scale': 0.0001,
                       'target-rms': 1.0,
                       'ng-affine-options': '',
                       'ng-linear-options': '',    # only affects bottleneck layers.
                       'dropout-proportion': 0.5,  # dropout-proportion only
                                                   # affects layers with
                                                   # 'dropout' in the name
                       'dropout-per-dim': False,  # if dropout-per-dim=true, the dropout
                                                  # mask is shared across time.
                       'dropout-per-dim-continuous':  False, # if you set this, it's
                                                    # like dropout-per-dim but with a
                                                    # continuous-valued (not zero-one) mask.
                       'add-log-stddev': False,
                       # the following are not really inspected by this level of
                       # code, just passed through to the affine component if
                       # their value is not ''.
                       'bias-stddev': '',
                       'l2-regularize': '',
                       'learning-rate-factor': '',
                       'max-change': 0.75 }

    def check_configs(self):
        if self.config['dim'] < 0:
            raise RuntimeError("dim has invalid value {0}".format(self.config['dim']))
        b = self.config['bottleneck-dim']
        if b >= 0 and (b >= self.config['dim'] or b == 0):
            raise RuntimeError("bottleneck-dim has an invalid value {0}".format(b))

        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if (self.config['learning-rate-factor'] != '' and
            self.config['learning-rate-factor'] <= 0.0):
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output is None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        # return something like: layer3.renorm
        return '{0}.{1}'.format(self.name, last_nonlinearity)

    def output_dim(self, auxiliary_output=None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
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
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, nonlinearities):
        output_dim = self.output_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        speaker_input = self.config['speaker-id']
        speaker_count = self.config['speaker-count']
        col_num = self.config['col-num']
        blhuc_activation = self.config['blhuc-activation']
        blhuc_mean_options = self.config['ng-blhuc-mean-options']
        blhuc_std_options = self.config['ng-blhuc-std-options']
        prior_mean_options = self.config['ng-prior-mean-options']
        prior_std_options = self.config['ng-prior-std-options']
        tied_std_scale = self.config['tied-std-scale']
        KL_scale = self.config['KL-scale']

        affine_options = self.config['ng-affine-options']
        for opt_name in [ 'max-change', 'learning-rate-factor',
                          'bias-stddev', 'l2-regularize' ]:
            value = self.config[opt_name]
            if value != '':
                affine_options += ' {0}={1}'.format(opt_name, value)

        # The output of the affine component needs to have one dimension fewer in order to
        # get the required output dim, if the final 'renorm' component has 'add-log-stddev' set
        # (since in that case it increases the dimension by one).
        if self.config['add-log-stddev']:
            output_dim -= 1
            if not self.layer_type.split('-')[-2] == "renorm":
                raise RuntimeError("add-log-stddev cannot be true unless "
                                   "there is a final 'renorm' component.")

        configs = []
        cur_dim = input_dim
        cur_node = input_desc

        # First the affine node (or linear then affine, if bottleneck).
        if self.config['bottleneck-dim'] > 0:
            # The 'bottleneck-dim' option is deprecated and may eventually be
            # removed.  Best to use tdnnf-layer if you want factorized TDNNs.

            # This is the bottleneck case (it doesn't necessarily imply we
            # will be using the features from the bottleneck; it's just a factorization
            # of the matrix into two pieces without a nonlinearity in between).
            # We don't include the l2-regularize option because it's useless
            # given the orthonormality constraint.
            linear_options = self.config['ng-linear-options']
            for opt_name in [ 'max-change', 'learning-rate-factor' ]:
                value = self.config[opt_name]
                if value != '':
                    linear_options += ' {0}={1}'.format(opt_name, value)

            bottleneck_dim = self.config['bottleneck-dim']
            # note: by default the LinearComponent uses natural gradient.
            line = ('component name={0}.linear type=LinearComponent '
                    'input-dim={1} orthonormal-constraint=1.0 output-dim={2} {3}'
                    ''.format(self.name, input_dim, bottleneck_dim, linear_options))
            configs.append(line)
            line = ('component-node name={0}.linear component={0}.linear input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.linear'.format(self.name)
            cur_dim = bottleneck_dim


        line = ('component name={0}.affine type=NaturalGradientAffineComponent'
                ' input-dim={1} output-dim={2} {3}'
                ''.format(self.name, cur_dim, output_dim, affine_options))
        configs.append(line)
        line = ('component-node name={0}.affine component={0}.affine input={1}'
                ''.format(self.name, cur_node))
        configs.append(line)
        cur_node = '{0}.affine'.format(self.name)

        for i, nonlinearity in enumerate(nonlinearities):
            if nonlinearity == 'relu':
                line = ('component name={0}.{1} type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  self_repair_scale))

            elif nonlinearity == 'renorm':
                add_log_stddev = "false"
                if i == len(nonlinearities) - 1:
                    add_log_stddev = ("true" if self.config['add-log-stddev']
                                      else "false")
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ' add-log-stddev={4}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  target_rms, add_log_stddev))

            elif nonlinearity == 'batchnorm':
                line = ('component name={0}.{1}'
                        ' type=BatchNormComponent dim={2} target-rms={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                                  target_rms))

            elif nonlinearity == 'so':
                line = ('component name={0}.{1}'
                        ' type=ScaleAndOffsetComponent dim={2} max-change=0.5 '
                        ''.format(self.name, nonlinearity, output_dim))

            elif nonlinearity == 'dropout':
                if not (self.config['dropout-per-dim'] or
                        self.config['dropout-per-dim-continuous']):
                    line = ('component name={0}.{1} type=DropoutComponent '
                            'dim={2} dropout-proportion={3}'.format(
                                self.name, nonlinearity, output_dim,
                                self.config['dropout-proportion']))
                else:
                    continuous_opt='continuous=true' if self.config['dropout-per-dim-continuous'] else ''

                    line = ('component name={0}.dropout type=GeneralDropoutComponent '
                            'dim={1} dropout-proportion={2} {3}'.format(
                                self.name, output_dim, self.config['dropout-proportion'],
                                continuous_opt))
            elif nonlinearity == 'blhuc1':
                line = ('component name={0}.BLHUC.rand type=NormalRandComponent '
                        'input-dim=1 output-dim={1} rand-per-frame=false test-mode=false'.format(self.name, output_dim))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.rand component={0}.BLHUC.rand input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.mean type=LinearSelectColComponent '
                        'input-dim=1 output-dim={1} col-num={2} {3}'.format(self.name, output_dim, col_num, blhuc_mean_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.mean component={0}.BLHUC.mean input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.std1 type=LinearSelectColComponent '
                        'input-dim=1 output-dim=1 col-num={1} {2}'.format(self.name, col_num, blhuc_std_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std1 component={0}.BLHUC.std1 input={1}'.format(self.name, speaker_input))
                configs.append(line)
                line = ('component name={0}.BLHUC.std2 type=NoOpComponent '
                        'dim=1 backprop-scale={1}'.format(self.name, tied_std_scale))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std2 component={0}.BLHUC.std2 input={0}.BLHUC.std1'.format(self.name))
                configs.append(line)
                line = ('component name={0}.BLHUC.std type=CopyNComponent '
                        'input-dim=1 output-dim={1}'.format(self.name, output_dim))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.std component={0}.BLHUC.std input={0}.BLHUC.std2'.format(self.name))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.productstd type=ElementwiseProductComponent input-dim={1} output-dim={2}'.format(self.name, 2*output_dim, output_dim))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.productstd component={0}.BLHUC.productstd input=Append({0}.BLHUC.rand, {0}.BLHUC.std)'.format(self.name))
                configs.append(line)
                line = ('component name={0}.BLHUC.summean type=SumBlockComponent input-dim={1} output-dim={2}'.format(self.name, 2*output_dim, output_dim))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.summean component={0}.BLHUC.summean input=Append({0}.BLHUC.productstd, {0}.BLHUC.mean)'.format(self.name))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.prior_mean type=ConstantFunctionComponent '
                        'input-dim=1 output-dim={1} {2}'.format(self.name, output_dim, prior_mean_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.prior_mean component={0}.BLHUC.prior_mean input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.prior_std type=ConstantFunctionComponent '
                        'input-dim=1 output-dim={1} {2}'.format(self.name, output_dim, prior_std_options))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.prior_std component={0}.BLHUC.prior_std input={1}'.format(self.name, speaker_input))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.KL type=KLGaussianComponent output-dim=1 input-dim={1} scale=1.0 input-frame-scale=true inv-frame-scale=true output-sum=true has-output=false'.format(self.name, output_dim*4+1))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.KL component={0}.BLHUC.KL input=Append({0}.BLHUC.mean, {0}.BLHUC.std, {0}.BLHUC.prior_mean, {0}.BLHUC.prior_std, {1})'.format(self.name, speaker_count))
                configs.append(line)
                line = ('component name={0}.BLHUC.KLmin type=MinValueComponent dim=1 scale={1}'.format(self.name, KL_scale))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.KLmin component={0}.BLHUC.KLmin input={0}.BLHUC.KL'.format(self.name))
                configs.append(line)
                
                if blhuc_activation == 'sigmoid':
                    line = ('component name={0}.BLHUC.sigmoid type=SigmoidComponent dim={1} self-repair-scale=0'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.sigmoid component={0}.BLHUC.sigmoid input={0}.BLHUC.summean'.format(self.name))
                    configs.append(line)
                elif blhuc_activation == 'linear':
                    line = ('component name={0}.BLHUC.noop type=NoOpComponent dim={1}'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.noop component={0}.BLHUC.noop input={0}.BLHUC.summean'.format(self.name))
                    configs.append(line)
                elif blhuc_activation == 'exp':
                    line = ('component name={0}.BLHUC.exp type=ExpComponent dim={1} self-repair-scale=0'.format(self.name, output_dim))
                    configs.append(line)
                    line = ('component-node name={0}.BLHUC.exp component={0}.BLHUC.exp input={0}.BLHUC.summean'.format(self.name))
                    configs.append(line)
                line = ('component name={0}.product type=ElementwiseProductComponent output-dim={1} input-dim={2}'.format(self.name, output_dim, 2*output_dim))
                configs.append(line)
                if blhuc_activation == 'sigmoid':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, Scale(2.0, {0}.BLHUC.sigmoid))'.format(self.name, cur_node))
                elif blhuc_activation == 'linear':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, {0}.BLHUC.noop)'.format(self.name, cur_node))
                elif blhuc_activation == 'exp':
                    line = ('component-node name={0}.product component={0}.product input=Append({1}, {0}.BLHUC.exp)'.format(self.name, cur_node))
                configs.append(line)
                
                line = ('component name={0}.BLHUC.withKL type=NoOpComponent dim={1}'.format(self.name, output_dim+1))
                configs.append(line)
                line = ('component-node name={0}.BLHUC.withKL component={0}.BLHUC.withKL input=Append({0}.product, {0}.BLHUC.KLmin)'.format(self.name))
                configs.append(line)
                line = ('dim-range-node name={0}.BLHUC.final input-node={0}.BLHUC.withKL dim={1} dim-offset=0'.format(self.name, output_dim))
                configs.append(line)
                
                nonlinearity='BLHUC.final'
            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   .format(nonlinearity))

            if nonlinearity != 'BLHUC.final':
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))

                configs.append(line)
            cur_node = '{0}.{1}'.format(self.name, nonlinearity)
        return configs


class XconfigTdnnfBLHUCLayer(XconfigLayerBase):

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "tdnnf-blhuc-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'dim':-1,
                       'bottleneck-dim':-1,
                       'speaker-id': '[-1]',
                       'speaker-count': '[-1]',
                       'col-num': -1,
                       'blhuc-activation': 'sigmoid', # 0: 2sigmoid; 1: linear; 2: exp
                       'ng-blhuc-mean-options': '',
                       'ng-blhuc-std-options': '',
                       'ng-prior-mean-options': '',
                       'ng-prior-std-options': '',
                       'tied-std-scale': 1.0,
                       'KL-scale': 0.0001,
                       'bypass-scale':0.66,
                       'dropout-proportion':-1.0,
                       'time-stride':1,
                       'l2-regularize':0.0,
                       'max-change': 0.75,
                       'self-repair-scale': 1.0e-05}

    def set_derived_configs(self):
        pass

    def check_configs(self):
        if self.config['bottleneck-dim'] <= 0:
            raise RuntimeError("bottleneck-dim must be set and >0.")
        if self.config['dim'] <= self.config['bottleneck-dim']:
            raise RuntimeError("dim must be greater than bottleneck-dim")

        dropout = self.config['dropout-proportion']
        if dropout != -1.0 and not (dropout >= 0.0 and dropout < 1.0):
            raise RuntimeError("invalid value for dropout-proportion")

        if abs(self.config['bypass-scale']) > 1.0:
            raise RuntimeError("bypass-scale has invalid value")

        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        if output_dim != input_dim and self.config['bypass-scale'] != 0.0:
            raise RuntimeError('bypass-scale is nonzero but output-dim != input-dim: {0} != {1}'
                               ''.format(output_dim, input_dim))


    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_component = ''
        if self.config['bypass-scale'] != 0.0:
            # the no-op component is used to cache something that we don't want
            # to have to recompute.
            output_component = 'noop'
        elif self.config['dropout-proportion'] != -1.0:
            output_component = 'dropout'
        else:
            output_component = 'batchnorm'
        return '{0}.{1}'.format(self.name, output_component)


    def output_dim(self, auxiliary_output=None):
        return self.config['dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans


    def _generate_config(self):
        configs = []
        name = self.name
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        speaker_input = self.config['speaker-id']
        speaker_count = self.config['speaker-count']
        col_num = self.config['col-num']
        blhuc_activation = self.config['blhuc-activation']
        blhuc_mean_options = self.config['ng-blhuc-mean-options']
        blhuc_std_options = self.config['ng-blhuc-std-options']
        prior_mean_options = self.config['ng-prior-mean-options']
        prior_std_options = self.config['ng-prior-std-options']
        tied_std_scale = self.config['tied-std-scale']
        KL_scale = self.config['KL-scale']
        bottleneck_dim = self.config['bottleneck-dim']
        bypass_scale = self.config['bypass-scale']
        dropout_proportion = self.config['dropout-proportion']
        time_stride = self.config['time-stride']
        if time_stride != 0:
            time_offsets1 = '{0},0'.format(-time_stride)
            time_offsets2 = '0,{0}'.format(time_stride)
        else:
            time_offsets1 = '0'
            time_offsets2 = '0'
        l2_regularize = self.config['l2-regularize']
        max_change = self.config['max-change']
        self_repair_scale = self.config['self-repair-scale']

        # The first linear layer, from input-dim (spliced x2) to bottleneck-dim
        configs.append('component name={0}.linear type=TdnnComponent input-dim={1} '
                       'output-dim={2} l2-regularize={3} max-change={4} use-bias=false '
                       'time-offsets={5} orthonormal-constraint=-1.0'.format(
                           name, input_dim, bottleneck_dim, l2_regularize,
                           max_change, time_offsets1))
        configs.append('component-node name={0}.linear component={0}.linear '
                       'input={1}'.format(name, input_descriptor))

        # The affine layer, from bottleneck-dim (spliced x2) to output-dim
        configs.append('component name={0}.affine type=TdnnComponent '
                       'input-dim={1} output-dim={2} l2-regularize={3} max-change={4} '
                       'time-offsets={5}'.format(
                           name, bottleneck_dim, output_dim, l2_regularize,
                           max_change, time_offsets2))
        configs.append('component-node name={0}.affine component={0}.affine '
                       'input={0}.linear'.format(name))

        # The ReLU layer
        configs.append('component name={0}.relu type=RectifiedLinearComponent dim={1} '
                       'self-repair-scale={2}'.format(
                           name, output_dim, self_repair_scale))
        configs.append('component-node name={0}.relu component={0}.relu '
                       'input={0}.affine'.format(name))

        configs.append('component name={0}.BLHUC.mean type=LinearSelectColComponent input-dim=1 output-dim={1} col-num={2} {3}'.format(name, output_dim, col_num, blhuc_mean_options))
        configs.append('component-node name={0}.BLHUC.mean component={0}.BLHUC.mean input={1}'.format(name, speaker_input))
                
        configs.append('component name={0}.BLHUC.std1 type=LinearSelectColComponent input-dim=1 output-dim=1 col-num={1} {2}'.format(name, col_num, blhuc_std_options))
        configs.append('component-node name={0}.BLHUC.std1 component={0}.BLHUC.std1 input={1}'.format(name, speaker_input))
        configs.append('component name={0}.BLHUC.std2 type=NoOpComponent dim=1 backprop-scale={1}'.format(name, tied_std_scale))
        configs.append('component-node name={0}.BLHUC.std2 component={0}.BLHUC.std2 input={0}.BLHUC.std1'.format(name))
        configs.append('component name={0}.BLHUC.std type=CopyNComponent input-dim=1 output-dim={1}'.format(name, output_dim))
        configs.append('component-node name={0}.BLHUC.std component={0}.BLHUC.std input={0}.BLHUC.std2'.format(name))
                
        configs.append('component name={0}.BLHUC.prior_mean type=ConstantFunctionComponent input-dim=1 output-dim={1} {2}'.format(name, output_dim, prior_mean_options))
        configs.append('component-node name={0}.BLHUC.prior_mean component={0}.BLHUC.prior_mean input={1}'.format(name, speaker_input))
                
        configs.append('component name={0}.BLHUC.prior_std type=ConstantFunctionComponent input-dim=1 output-dim={1} {2}'.format(name, output_dim, prior_std_options))
        configs.append('component-node name={0}.BLHUC.prior_std component={0}.BLHUC.prior_std input={1}'.format(name, speaker_input))
                
        configs.append('component name={0}.BLHUC.vec type=BayesVecKLGaussianComponent output-dim={1} input-dim={2} KL-scale={3} input-frame-scale=true inv-frame-scale=true rand-per-frame=false KL-output=false test-mode=false'.format(name, output_dim, output_dim*4+1, KL_scale))
        configs.append('component-node name={0}.BLHUC.vec component={0}.BLHUC.vec input=Append({0}.BLHUC.mean, {0}.BLHUC.std, {0}.BLHUC.prior_mean, {0}.BLHUC.prior_std, {1})'.format(name, speaker_count))
        
        if blhuc_activation == 'sigmoid':
            configs.append('component name={0}.BLHUC.sigmoid type=SigmoidComponent dim={1} self-repair-scale=0'.format(name, output_dim))
            configs.append('component-node name={0}.BLHUC.sigmoid component={0}.BLHUC.sigmoid input={0}.BLHUC.vec'.format(name))
        elif blhuc_activation == 'linear':
            configs.append('component name={0}.BLHUC.noop type=NoOpComponent dim={1}'.format(name, output_dim))
            configs.append('component-node name={0}.BLHUC.noop component={0}.BLHUC.noop input={0}.BLHUC.vec'.format(name))
        elif blhuc_activation == 'exp':
            configs.append('component name={0}.BLHUC.exp type=ExpComponent dim={1} self-repair-scale=0'.format(name, output_dim))
            configs.append('component-node name={0}.BLHUC.exp component={0}.BLHUC.exp input={0}.BLHUC.vec'.format(name))
        configs.append('component name={0}.product type=ElementwiseProductComponent output-dim={1} input-dim={2}'.format(name, output_dim, 2*output_dim))
        if blhuc_activation == 'sigmoid':
            configs.append('component-node name={0}.product component={0}.product input=Append({0}.relu, Scale(2.0, {0}.BLHUC.sigmoid))'.format(name))
        elif blhuc_activation == 'linear':
            configs.append('component-node name={0}.product component={0}.product input=Append({0}.relu, {0}.BLHUC.noop)'.format(name))
        elif blhuc_activation == 'exp':
            configs.append('component-node name={0}.product component={0}.product input=Append({0}.relu, {0}.BLHUC.exp)'.format(name))

        # The BatchNorm layer
        configs.append('component name={0}.batchnorm type=BatchNormComponent '
                       'dim={1}'.format(name, output_dim))
        configs.append('component-node name={0}.batchnorm component={0}.batchnorm '
                       'input={0}.product'.format(name))

        if dropout_proportion != -1:
            # This is not normal dropout.  It's dropout where the mask is shared
            # across time, and (thanks to continuous=true), instead of a
            # zero-or-one scale, it's a continuously varying scale whose
            # expected value is 1, drawn from a uniform distribution over an
            # interval of a size that varies with dropout-proportion.
            configs.append('component name={0}.dropout type=GeneralDropoutComponent '
                           'dim={1} dropout-proportion={2} continuous=true'.format(
                               name, output_dim, dropout_proportion))
            configs.append('component-node name={0}.dropout component={0}.dropout '
                           'input={0}.batchnorm'.format(name))
            cur_component_type = 'dropout'
        else:
            cur_component_type = 'batchnorm'

        if bypass_scale != 0.0:
            # Add a NoOpComponent to cache the weighted sum of the input and the
            # output.  We could easily have the output of the component be a
            # Descriptor like 'Append(Scale(0.66, tdnn1.batchnorm), tdnn2.batchnorm)',
            # but if we did that and you used many of this component in sequence,
            # the weighted sums would have more and more terms as you went deeper
            # in the network.
            configs.append('component name={0}.noop type=NoOpComponent '
                           'dim={1}'.format(name, output_dim))
            configs.append('component-node name={0}.noop component={0}.noop '
                           'input=Sum(Scale({1}, {2}), {0}.{3})'.format(
                               name, bypass_scale, input_descriptor,
                               cur_component_type))

        return configs
