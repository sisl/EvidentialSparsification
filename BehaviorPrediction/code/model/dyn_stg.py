import numpy as np
import torch
from model.node_model import MultimodalGenerativeCVAE


class SpatioTemporalGraphCVAEModel(object):
    def __init__(self, robot_node, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(SpatioTemporalGraphCVAEModel, self).__init__()
        self.hyperparams = hyperparams
        self.robot_node = robot_node
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.dims = self.hyperparams['dims']
        self.pred_state = self.hyperparams['pred_state']
        self.pred_state_indices = np.arange(self.state.index(self.pred_state) * len(self.dims),
                                            self.state.index(self.pred_state) * len(self.dims) + len(self.dims))

    def set_scene_graph(self, repr_scene):
        self.node_models_dict.clear()

        edge_types = repr_scene.get_edge_types()

        for node_type in repr_scene.type_enum:
            self.node_models_dict[node_type] = MultimodalGenerativeCVAE(node_type,
                                                                        self.model_registrar,
                                                                        self.robot_node,
                                                                        self.hyperparams,
                                                                        self.device,
                                                                        edge_types,
                                                                        log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self):
        for node in self.node_models_dict:
            self.node_models_dict[node].step_annealers()

    def get_input(self, scene, timesteps, node_type, min_future_timesteps):
        inputs = list()
        first_history_indices = list()
        nodes = list()
        node_scene_graph_batched = list()
        timesteps_in_scene = list()
        nodes_per_ts = scene.present_nodes(timesteps,
                                           type=node_type,
                                           min_history_timesteps=self.min_hl,
                                           min_future_timesteps=min_future_timesteps)

        # Get Inputs for each node present in Scene
        for timestep in timesteps:
            if timestep in nodes_per_ts.keys():
                present_nodes = nodes_per_ts[timestep]
                timestep_range = np.array([timestep - self.max_hl, timestep + min_future_timesteps])
                scene_graph_t = scene.get_scene_graph(timestep,
                                                      edge_radius=self.hyperparams['edge_radius'],
                                                      dims=self.dims,
                                                      edge_addition_filter_l=len(self.hyperparams['edge_addition_filter']),
                                                      edge_removal_filter_l=len(self.hyperparams['edge_removal_filter']))

                for node in present_nodes:
                    timesteps_in_scene.append(timestep)
                    input = node.get(timestep_range, self.state, self.dims)
                    first_history_index = (self.max_hl - node.history_points_at(timestep)).clip(0)
                    inputs.append(input)
                    first_history_indices.append(first_history_index)
                    nodes.append(node)

                    node_scene_graph_batched.append((node, scene_graph_t))

        return inputs, first_history_indices, timesteps_in_scene, node_scene_graph_batched, nodes

    def train_loss(self, scene, timesteps):
        losses = list()
        for node_type in scene.type_enum:
            # Get Input data for node type and given timesteps
            (inputs,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched, _) = self.get_input(scene, timesteps, node_type, self.ph)

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            # Standardize
            inputs_st = scene.standardize(inputs, self.state, self.dims)

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()
            labels = inputs[:, :, self.pred_state_indices]
            labels_st = inputs_st[:, :, self.pred_state_indices]

            # Run forward pass
            model = self.node_models_dict[node_type]
            loss = model.train_loss(inputs,
                                    inputs_st,
                                    first_history_indices,
                                    labels,
                                    labels_st,
                                    scene,
                                    node_scene_graph_batched,
                                    timestep=self.max_hl,
                                    timesteps_in_scene=timesteps_in_scene,
                                    prediction_horizon=self.ph)

            losses.append(loss)

        mean_loss = torch.mean(torch.stack(losses)) if len(losses) > 0 else None
        return mean_loss

    def eval_loss(self, scene, timesteps):
        nll_q_is_values = list()
        nll_p_values = list()
        nll_exact_values = list()
        nll_sampled_values = list()
        for node_type in scene.type_enum:
            # Get Input data for node type and given timesteps
            (inputs,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched, _) = self.get_input(scene, timesteps, node_type, self.ph)

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            # Standardize
            inputs_st = scene.standardize(inputs, self.state, self.dims)

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()
            labels = inputs[:, :, self.pred_state_indices]
            labels_st = inputs_st[:, :, self.pred_state_indices]

            # Run forward pass
            model = self.node_models_dict[node_type]
            (nll_q_is, nll_p, nll_exact, nll_sampled) = model.eval_loss(inputs,
                                                                        inputs_st,
                                                                        first_history_indices,
                                                                        labels,
                                                                        labels_st,
                                                                        scene,
                                                                        node_scene_graph_batched,
                                                                        timestep=self.max_hl,
                                                                        timesteps_in_scene=timesteps_in_scene,
                                                                        prediction_horizon=self.ph)

            if nll_q_is is not None:
                nll_q_is_values.append(nll_q_is)
                nll_p_values.append(nll_p)
                nll_exact_values.append(nll_exact)
                nll_sampled_values.append(nll_sampled)

        (nll_q_is, nll_p, nll_exact, nll_sampled) = (torch.mean(torch.stack(nll_q_is_values)),
                                                     torch.mean(torch.stack(nll_p_values)),
                                                     torch.mean(torch.stack(nll_exact_values)),
                                                     torch.mean(torch.stack(nll_sampled_values)))
        return nll_q_is.cpu().numpy(), nll_p.cpu().numpy(), nll_exact.cpu().numpy(), nll_sampled.cpu().numpy()

    def predict(self, scene, timesteps, ph, num_samples, min_future_timesteps=0, most_likely_z=False, sample_probs=None):
        predictions_dict = {}
        for node_type in scene.type_enum:
            # Get Input data for node type and given timesteps
            (inputs,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched,
             nodes) = self.get_input(scene, timesteps, node_type, min_future_timesteps)

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            # Standardize
            inputs_st = scene.standardize(inputs, self.state, self.dims)

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()

            # Run forward pass
            model = self.node_models_dict[node_type]
            predictions = model.predict(inputs,
                                        inputs_st,
                                        first_history_indices,
                                        scene,
                                        node_scene_graph_batched,
                                        timestep=self.max_hl,
                                        timesteps_in_scene=timesteps_in_scene,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        most_likely_z=most_likely_z,
                                        sample_probs=sample_probs
                                        )
            predictions_uns = scene.unstandardize(predictions.detach().numpy(), [self.pred_state], self.dims)

            print(predictions_uns.shape)
            # Assign predictions to node
            for i, ts in enumerate(timesteps_in_scene):
                if not  ts in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = predictions_uns[:, i]

        return predictions_dict

