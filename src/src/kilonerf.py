import torch
import torch.nn as nn

class KiloNeRF(nn.Module):
    """
    Neural Radiance Fields with Thousands of Tiny MLPs
    """

    def __init__(self, num_networks, num_position_channels, num_direction_channels, num_output_channels, hidden_layer_size, num_hidden_layers, refeed_position_index=None, late_feed_direction=False,
        direction_layer_size=None, nonlinearity='relu', nonlinearity_initalization='pass_leaky_relu', use_single_net=False, linear_implementation='bmm', use_same_initialization_for_all_networks=False,
        network_rng_seed=None, weight_initialization_method='kaiming_uniform', bias_initialization_method='standard', alpha_rgb_initalization='updated_yenchenlin', use_hard_parameter_sharing_for_color=False,
        view_dependent_dropout_probability=-1, use_view_independent_color=False):
        super(KiloNeRF, self).__init__()
        
        self.num_networks = num_networks
        self.num_position_channels = num_position_channels
        self.num_direction_channels = num_direction_channels
        self.num_output_channels = num_output_channels
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.refeed_position_index = refeed_position_index
        self.late_feed_direction = late_feed_direction
        self.direction_layer_size = direction_layer_size
        self.nonlinearity = nonlinearity
        self.nonlinearity_initalization = nonlinearity_initalization # 'pass_leaky_relu', 'pass_actual_nonlinearity'
        self.use_single_net = use_single_net
        self.linear_implementation = linear_implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        self.alpha_rgb_initalization = alpha_rgb_initalization # 'updated_yenchenlin', 'pass_actual_nonlinearity'
        self.use_hard_parameter_sharing_for_color = use_hard_parameter_sharing_for_color
        self.view_dependent_dropout_probability = view_dependent_dropout_probability
        self.use_view_independent_color = use_view_independent_color
        
        nonlinearity_params = {}
        if nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()
        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        if nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        if nonlinearity == 'sine':
            nonlinearity_params = {'w0': 30., 'c': 6., 'is_first': True}
            self.activation = Sine(nonlinearity_params['w0'])
            
        # TODO: weight_initalization_method and bias_initalization_method are being ignored
        def linear_layer(in_features, out_features, actual_nonlinearity, use_hard_parameter_sharing=False):
            if self.nonlinearity_initalization == 'pass_actual_nonlinearity': # proper way of doing things
                passed_nonlinearity = actual_nonlinearity 
            elif self.nonlinearity_initalization == 'pass_leaky_relu': # to reproduce the old behaviour (doesn't make a lot of sense though)
                passed_nonlinearity = 'leaky_relu'
            if not use_hard_parameter_sharing:
                return MultiNetworkLinear(self.num_networks, in_features, out_features,
                    nonlinearity=passed_nonlinearity, nonlinearity_params=nonlinearity_params, implementation=linear_implementation,
                    use_same_initialization_for_all_networks=use_same_initialization_for_all_networks, network_rng_seed=network_rng_seed)
            else:
                print('Using hard parameter sharing')
                return SharedLinear(in_features, out_features, bias=True, nonlinearity=passed_nonlinearity)

        if self.late_feed_direction:
            self.pts_linears = [linear_layer(self.num_position_channels, self.hidden_layer_size, self.nonlinearity)]
            nonlinearity_params = nonlinearity_params.copy().update({'is_first': False})
            for i in range(self.num_hidden_layers - 1):
                if i == self.refeed_position_index:
                    new_layer = linear_layer(self.hidden_layer_size + self.num_position_channels, self.hidden_layer_size, self.nonlinearity)
                else:
                    new_layer = linear_layer(self.hidden_layer_size, self.hidden_layer_size, self.nonlinearity)
                self.pts_linears.append(new_layer)
            self.pts_linears = nn.ModuleList(self.pts_linears)
            self.direction_layer = linear_layer(self.num_direction_channels + self.hidden_layer_size, self.direction_layer_size, self.nonlinearity, self.use_hard_parameter_sharing_for_color)
            
            if self.use_view_independent_color:
                feature_output_size = self.hidden_layer_size + 4 # + RGBA
            else:
                feature_output_size = self.hidden_layer_size
            self.feature_linear = linear_layer(self.hidden_layer_size, feature_output_size, 'linear')
            # In the updated yenchenlin implementation which follows now closely the original tensorflow implementation
            # 'linear' is passed to these two layers, but it also makes sense to pass the actual nonlinearites here
            if not self.use_view_independent_color:
                self.alpha_linear = linear_layer(self.hidden_layer_size, 1, 'linear' if self.alpha_rgb_initalization == 'updated_yenchenlin' else 'relu')
            self.rgb_linear = linear_layer(self.direction_layer_size, 3, 'linear' if self.alpha_rgb_initalization == 'updated_yenchenlin' else 'sigmoid',
                self.use_hard_parameter_sharing_for_color)
                
            self.view_dependent_parameters = list(self.direction_layer.parameters()) + list(self.rgb_linear.parameters()) # needed for L2 regularization only on the view-dependent part of the network
            
            if self.view_dependent_dropout_probability > 0:
                self.dropout_after_feature = nn.Dropout(self.view_dependent_dropout_probability)
                self.dropout_after_direction_layer = nn.Dropout(self.view_dependent_dropout_probability)
            
        else:
            layers = [linear_layer(self.num_position_channels + self.num_direction_channels, self.hidden_layer_size), self.activation]
            nonlinearity_params = nonlinearity_params.copy().update({'is_first': False})
            for _ in range(self.num_hidden_layers): # TODO: should be also self.num_hidden_layers - 1
                layers += [linear_layer(self.hidden_layer_size, self.hidden_layer_size), self.activation]
            layers += [linear_layer(self.hidden_layer_size, self.num_output_channels)]
            self.layers = nn.Sequential(*layers)
    
    # needed for fused kernel
    def serialize_params(self):
        # fused kernel expects IxO matrix instead of OxI matrix
        def process_weight(w):
            return w.reshape(self.num_networks, -1)
    
        self.serialized_params = []
        for l in self.pts_linears:
            self.serialized_params += [l.bias, process_weight(l.weight)]
            
        self.serialized_params.append(torch.cat([self.alpha_linear.bias, self.feature_linear.bias], dim=1))
        self.serialized_params.append(process_weight(torch.cat([self.alpha_linear.weight, self.feature_linear.weight], dim=2)))
        for l in [self.direction_layer, self.rgb_linear]:
            self.serialized_params += [l.bias, process_weight(l.weight)]
        self.serialized_params = torch.cat(self.serialized_params, dim=1).contiguous()

    # random_directions will be used for regularizing the view-independent color
    def forward(self, x, batch_size_per_network=None, random_directions=None):
        if self.late_feed_direction:
            if isinstance(x, list):
                positions, directions = x
                # frees memory of inputs
                x[0] = None 
                x[1] = None
            else:
                positions, directions = torch.split(x, [self.num_position_channels, self.num_direction_channels], dim=-1)
            h = positions
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h, batch_size_per_network)
                PerfMonitor.add('pts_linears ' + str(i), ['network query', 'matmul'])
                h = self.activation(h)
                PerfMonitor.add('activation ' + str(i), ['network query', 'matmul'])
                if i == self.refeed_position_index:
                    h = torch.cat([positions, h], -1)
                    PerfMonitor.add('cat[positions, h]', ['network query', ])
            del positions
            if not self.use_view_independent_color:
                alpha = self.alpha_linear(h, batch_size_per_network)
                PerfMonitor.add('alpha_linear', ['network query', 'matmul'])
            feature = self.feature_linear(h, batch_size_per_network) # TODO: investigate why they don't use an activation function on top of feature layer!
            if self.view_dependent_dropout_probability > 0:
                feature =  self.dropout_after_feature(feature)
            if self.use_view_independent_color:
               rgb_view_independent, alpha, feature = torch.split(feature, [3, 1, self.hidden_layer_size], dim=-1)
            PerfMonitor.add('feature_linear', ['network query', 'matmul'])
            del h
            
            # Regularizing the view-independent color to be the mean of view-dependent colors sampled at some random directions
            if random_directions is not None:
                assert self.use_view_independent_color == True, 'this regularization only makes sense if we output a view-independent color'
                num_random_directions = random_directions.size(0)
                batch_size = feature.size(0)
                feature_size = feature.size(1)
                feature = feature.repeat(1, num_random_directions + 1).view(-1, feature_size)
                random_directions = random_directions.repeat(batch_size, 1).view(batch_size, num_random_directions, -1)
                directions = torch.cat([directions.unsqueeze(1), random_directions], dim=1).view(batch_size * (num_random_directions + 1), -1)
                batch_size_per_network = (num_random_directions + 1) * batch_size_per_network

            
            # View-dependent part of the network:
            h = torch.cat([feature, directions], -1)
            PerfMonitor.add('cat[feature, directions]', ['network query'])
            del feature
            del directions
            h = self.direction_layer(h, batch_size_per_network)
            PerfMonitor.add('direction_linear', ['network query', 'matmul'])
            h = self.activation(h)
            if self.view_dependent_dropout_probability > 0:
                h = self.dropout_after_direction_layer(h)
            PerfMonitor.add('direction activation', ['network query'])
            rgb = self.rgb_linear(h, batch_size_per_network)
            PerfMonitor.add('rgb_linear', ['network query', 'matmul'])
            del h

            if self.use_view_independent_color:
                if random_directions is None:
                    rgb = rgb + rgb_view_independent
                else:
                    mean_rgb = rgb.view(batch_size, num_random_directions + 1, 3)
                    mean_rgb = mean_rgb + rgb_view_independent.unsqueeze(1)
                    rgb = mean_rgb[:, 0]
                    mean_rgb = mean_rgb.mean(dim=1)
                    mean_regularization_term = torch.abs(mean_rgb - rgb_view_independent).mean()
                    del mean_rgb
                del rgb_view_independent
                PerfMonitor.add('rgb + rgb_view_independent', ['network query'])
                
            result = torch.cat([rgb, alpha], -1)
            PerfMonitor.add('cat[rgb, alpha]', ['network query'])
            
            if random_directions is not None:
                return result, mean_regularization_term
            else:
                return result
        else:
            return self.layers(x)
            

    def extract_single_network(self, network_index):
        single_network = MultiNetwork(1, self.num_position_channels, self.num_direction_channels, self.num_output_channels,
            self.hidden_layer_size, self.num_hidden_layers, self.refeed_position_index, self.late_feed_direction,
            self.direction_layer_size, self.nonlinearity, self.nonlinearity_initalization, self.use_single_net,
            use_hard_parameter_sharing_for_color=self.use_hard_parameter_sharing_for_color,
            view_dependent_dropout_probability=self.view_dependent_dropout_probability,
            use_view_independent_color=self.use_view_independent_color)
      
        multi_linears, multi_shared_linears = extract_linears(self)
        single_linears, single_shared_linears = extract_linears(single_network)
        with torch.no_grad():
            for single_linear, multi_linear in zip(single_linears, multi_linears):
                single_linear.weight.data[0] = multi_linear.weight.data[network_index]
                single_linear.bias.data[0] = multi_linear.bias.data[network_index]
                   
            for single_shared_linear, multi_shared_linear in zip(single_shared_linears, multi_shared_linears):
                single_shared_linear.weight.data = multi_shared_linear.weight.data
                single_shared_linear.bias.data = multi_shared_linear.bias.data
            
        return single_network
    
    # Just for the unit test
    def _extract_single_network(self, network_index):
        def copy_parameters(network_index, linear, multi_network_linear):
            with torch.no_grad():
                linear.weight.data[:] = multi_network_linear.weight.data[network_index]
                linear.bias.data[:] = multi_network_linear.bias.data[network_index]
                
        input_layer = nn.Linear(self.num_input_channels, self.hidden_layer_size)
        layer_index = 0
        copy_parameters(network_index, input_layer, self.layers[layer_index])
        single_network_layers = [input_layer, nn.ReLU()]
        layer_index = 2
        for _ in range(self.num_hidden_layers):
            hidden_layer = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            copy_parameters(network_index, hidden_layer, self.layers[layer_index])
            single_network_layers += [hidden_layer, nn.ReLU()]
            layer_index += 2
        output_layer = nn.Linear(self.hidden_layer_size, self.num_output_channels)
        copy_parameters(network_index, output_layer, self.layers[layer_index])
        single_network_layers += [output_layer]
        return  nn.Sequential(*single_network_layers)