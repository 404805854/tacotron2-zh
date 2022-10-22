# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from scipy.stats import betabinom
import math
import numpy as np
from hparams import default as hparams
from tacotron2_common.utils import to_gpu, get_mask_from_lengths
from tacotron2_common.layers import ConvNorm, LinearNorm
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)

        self.attention_mode = hparams.attention_mode
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


# Default Attention (Location Sensitive Attention, LSA)
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, decoder_dim)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


# Gaussian Mixture Model Attention, GMM
class GMMAttention(nn.Module):
    def __init__(self, query_dim, attention_dim, kernel, delta_bias, sigma_bias):
        super(GMMAttention, self).__init__()
        self.query_dim = query_dim
        self.attention_dim = attention_dim
        self.kernel = kernel
        self.delta_bias = delta_bias
        self.sigma_bias = sigma_bias
        self.query_layer = LinearNorm(
            query_dim, attention_dim, bias=True, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 3 * self.kernel, bias=True)
        self.score_mask_value = 0.0
        torch.nn.init.constant_(self.v.linear_layer.bias[(1 * self.kernel):(2 * self.kernel)],
                                self.delta_bias)  # bias mean
        torch.nn.init.constant_(self.v.linear_layer.bias[(2 * self.kernel):(3 * self.kernel)],
                                self.sigma_bias)  # bias std

    def forward(self, query, memory, prev_mu, memory_time, mask=None):
        processed_query = self.v(torch.tanh(
            self.query_layer(query)))  # [B, 3*K]
        w_hat, delta_hat, sigma_hat = torch.chunk(processed_query, 3, dim=1)
        w = torch.softmax(w_hat, dim=1).unsqueeze(2)  # [B, k, 1]
        delta = F.softplus(delta_hat).unsqueeze(2)  # [B, k, 1]
        sigma = F.softplus(sigma_hat).unsqueeze(2)  # [B, k, 1]
        current_mu = prev_mu + delta
        z = math.sqrt(2 * math.pi) * sigma  # [B, k, 1]
        energies = w / z * \
            torch.exp(-0.5 * (memory_time - current_mu)
                      ** 2 / sigma ** 2)  # [B, K, N]
        alignments = torch.sum(energies, dim=1)  # [B, N]
        if mask is not None:
            alignments.masked_fill_(mask, self.score_mask_value)
        attention_context = torch.bmm(alignments.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, alignments, current_mu


# Forward Attention, FA
class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        fwd_shifted_alpha = F.pad(
            log_alpha[:, :-1], [1, 0], 'constant', self.score_mask_value)
        biased = torch.logsumexp(
            torch.cat([log_alpha.unsqueeze(2), fwd_shifted_alpha.unsqueeze(2)], 2), 2)

        log_alpha_new = biased + log_energy

        attention_weights = F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new


# Attention_base
class BahdanauAttention(nn.Module):
    """
    BahdanauAttention

    This attention is described in:
        D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate,"
        in International Conference on Learning Representation (ICLR), 2015.
        https://arxiv.org/abs/1409.0473
    """

    def __init__(self, query_dim, memory_dim, attn_dim, score_mask_value=-float("inf")):
        super(BahdanauAttention, self).__init__()

        # Query layer to project query to hidden representation
        # (query_dim -> attn_dim)
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, attn_dim, bias=False)

        # For computing alignment energies
        self.tanh = nn.Tanh()
        self.v = nn.Linear(attn_dim, 1, bias=False)

        # For computing weights
        self.score_mask_value = score_mask_value

    def forward(self, query, processed_memory, mask=None):
        """
        Get normalized attention weight

        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
            mask: (batch, max_time)

        Returns:
            alignment: [batch, max_time]
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)

        # Alignment energies
        alignment = self.get_energies(query, processed_memory)

        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Alignment probabilities (attention weights)
        alignment = self.get_probabilities(alignment)

        # (batch, max_time)
        return alignment

    def init_attention(self, processed_memory):
        # Nothing to do in the base module
        return

    def get_energies(self, query, processed_memory):
        """
        Compute the alignment energies
        """
        # Query (batch, 1, dim)
        processed_query = self.query_layer(query)

        processed_memory = self.memory_layer(processed_memory)

        # Alignment energies (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        # (batch, max_time)
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        """
        Compute the alignment probabilites (attention weights) from energies
        """
        return nn.Softmax(dim=1)(energies)


# Stepwise Monotonic Attention, SMA
class StepwiseMonotonicAttention(BahdanauAttention):
    """
    StepwiseMonotonicAttention (SMA)

    This attention is described in:
        M. He, Y. Deng, and L. He, "Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS,"
        in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2019, pp. 1293-1297.
        https://arxiv.org/abs/1906.00672

    See:
        https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
        https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
    """

    def __init__(self, query_dim, memory_dim, attn_dim, sigmoid_noise=2.0, score_mask_value=-float("inf")):
        """
        Args:
            sigmoid_noise: Standard deviation of pre-sigmoid noise.
                           Setting this larger than 0 will encourage the model to produce
                           large attention scores, effectively making the choosing probabilities
                           discrete and the resulting attention distribution one-hot.
        """
        super(StepwiseMonotonicAttention, self).__init__(
            query_dim, memory_dim, attn_dim, score_mask_value)

        self.alignment = None  # alignment in previous query time step
        self.sigmoid_noise = sigmoid_noise

    def init_attention(self, processed_memory):
        # Initial alignment with [1, 0, ..., 0]
        b, t, c = processed_memory.size()
        self.alignment = processed_memory.new_zeros(b, t)
        self.alignment[:, 0:1] = 1

    def stepwise_monotonic_attention(self, p_i, prev_alignment):
        """
        Compute stepwise monotonic attention
            - p_i: probability to keep attended to the last attended entry
            - Equation (8) in section 3 of the paper
        """
        pad = prev_alignment.new_zeros(prev_alignment.size(0), 1)
        alignment = prev_alignment * p_i + \
            torch.cat((pad, prev_alignment[:, :-1]
                      * (1.0 - p_i[:, :-1])), dim=1)
        return alignment

    def get_selection_probability(self, e, std):
        """
        Compute selecton/sampling probability `p_i` from energies `e`
            - Equation (4) and the tricks in section 2.2 of the paper
        """
        # Add Gaussian noise to encourage discreteness
        if self.training:
            noise = e.new_zeros(e.size()).normal_()
            e = e + noise * std

        # Compute selecton/sampling probability p_i
        # (batch, max_time)
        return torch.sigmoid(e)

    def get_probabilities(self, energies):
        # Selecton/sampling probability p_i
        p_i = self.get_selection_probability(energies, self.sigmoid_noise)

        # Stepwise monotonic attention
        alignment = self.stepwise_monotonic_attention(p_i, self.alignment)

        # (batch, max_time)
        self.alignment = alignment
        return alignment


# Dynamic Convolutional Attention, DCA
class DynamicConvolutionAttention(BahdanauAttention):
    def __init__(self, query_dim, memory_dim, attn_dim, static_channels=8, static_kernel_size=21,
                 dynamic_channels=8, dynamic_kernel_size=21, prior_length=11,
                 alpha=0.1, beta=0.9, score_mask_value=-float("inf")):
        super(DynamicConvolutionAttention, self).__init__(
            query_dim, memory_dim, attn_dim, score_mask_value)
        self.prior_length = prior_length
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size

        P = betabinom.pmf(np.arange(prior_length),
                          prior_length - 1, alpha, beta)

        self.register_buffer("P", torch.FloatTensor(P).flip(0))
        self.W = nn.Linear(query_dim, attn_dim)
        self.V = nn.Linear(
            attn_dim, dynamic_channels * dynamic_kernel_size, bias=False
        )
        self.F = nn.Conv1d(
            1,
            static_channels,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.U = nn.Linear(static_channels, attn_dim, bias=False)
        self.T = nn.Linear(dynamic_channels, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def init_attention(self, processed_memory):
        b, t, _ = processed_memory.size()
        self.alignment_pre = F.one_hot(torch.zeros(
            b, dtype=torch.long), t).float().cuda()

    def get_energies(self, query, processed_memory):
        query = query.squeeze(1)
        p = F.conv1d(
            F.pad(self.alignment_pre.unsqueeze(1),
                  (self.prior_length - 1, 0)), self.P.view(1, 1, -1)
        )
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(query)))
        g = F.conv1d(
            self.alignment_pre.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        g = g.view(query.size(0), self.dynamic_channels, -1).transpose(1, 2)

        f = self.F(self.alignment_pre.unsqueeze(1)).transpose(1, 2)

        e = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

        return e

    def get_probabilities(self, energies):
        # Current attention
        alignment = nn.Softmax(dim=1)(energies)

        # Update previous attention
        self.alignment_pre = alignment

        return alignment


# GMM_Evo Attention
class GMMAttentionEvo(BahdanauAttention):
    def __init__(self, query_dim, memory_dim, attn_dim, K=hparams.GMM_Evo_K, version=hparams.GMM_Evo_Version,
                 score_mask_value=1e-8):
        super(GMMAttentionEvo, self).__init__(
            query_dim, memory_dim, attn_dim, score_mask_value)
        self.gmm_version = version
        self.K = K  # num mixture
        self.eps = 1e-5
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, attn_dim, bias=True),
            nn.Tanh(),
            nn.Linear(attn_dim, 3 * K))

    def init_attention(self, processed_memory):
        # No need to initialize alignment
        # because GMM Attention is purely location based
        # it has nothing to do with memory and t-1's alignment

        # Initial mu_pre with all zeros
        b, t, c = processed_memory.size()
        self.mu_prev = processed_memory.data.new(b, self.K, 1).zero_()
        # j = torch.arange(0, processed_memory.size(1)).to(processed_memory.device)
        j = torch.arange(0, processed_memory.size(1)).float().cuda()
        self.j = j.view(1, 1, processed_memory.size(1))  # [1, 1, T]

    def get_energies(self, query, processed_memory):
        """
         Args:
            query: (batch, dim)
            processed_memory: (batch, max_time, dim)
        Returns:
            alignment: [batch, max_time]
        """
        # Intermediate parameters (in Table 1)
        interm_params = self.mlp(query).view(
            query.size(0), -1, self.K)  # [B, 3, K]
        omega_hat, delta_hat, sigma_hat = interm_params.chunk(
            3, dim=1)  # Tuple

        # Each [B, K]
        omega_hat = omega_hat.squeeze(1)
        delta_hat = delta_hat.squeeze(1)
        sigma_hat = sigma_hat.squeeze(1)

        # Convert intermediate parameters to final mixture parameters
        # Choose version V0/V1/V2
        # Formula from https://arxiv.org/abs/1910.10288
        if self.gmm_version == '0':
            sigma = (torch.sqrt(torch.exp(-sigma_hat) / 2) +
                     self.eps).unsqueeze(-1)  # [B, K, 1]
            delta = torch.exp(delta_hat).unsqueeze(-1)  # [B, K, 1]
            omega = torch.exp(omega_hat).unsqueeze(-1)  # [B, K, 1]
            Z = 1.0
        elif self.gmm_version == '1':
            sigma = (torch.sqrt(torch.exp(sigma_hat)) + self.eps).unsqueeze(-1)
            delta = torch.exp(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma ** 2)
        elif self.gmm_version == '2':
            sigma = (F.softplus(sigma_hat) + self.eps).unsqueeze(-1)
            delta = F.softplus(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma ** 2)

        mu = self.mu_prev + delta  # [B, K, 1]

        # Get alignment(phi in mathtype)
        alignment = omega / Z * \
            torch.exp(-(self.j - mu) ** 2 / (sigma ** 2) / 2)  # [B, K ,T]
        alignment = torch.sum(alignment, 1)  # [B, T]

        # Update mu_prev
        self.mu_prev = mu

        return alignment

    def get_probabilities(self, energies):
        return energies


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, encoder_n_convolutions,
                 symbols_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions - 1):
            conv_layer = nn.Sequential(
                ConvNorm(symbols_embedding_dim,
                         symbols_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(symbols_embedding_dim))
            convolutions.append(conv_layer)

        conv_layer = nn.Sequential(
            ConvNorm(symbols_embedding_dim,
                     hparams.encoder_embedding_dim,
                     kernel_size=encoder_kernel_size, stride=1,
                     padding=int((encoder_kernel_size - 1) / 2),
                     dilation=1, w_init_gain='relu'),
            nn.BatchNorm1d(hparams.encoder_embedding_dim))
        convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

        self.spk_embedding_tf_layer = LinearNorm(
            512, hparams.spk_embedding_dim, w_init_gain='tanh')

    @torch.jit.ignore
    def forward(self, x, input_lengths, spk_embeddings):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        spk_embeddings_trans = self.spk_embedding_tf_layer(
            spk_embeddings).unsqueeze(1).expand(-1, outputs.size(1), -1)

        final_outputs = torch.cat([outputs, spk_embeddings_trans], dim=-1)

        return final_outputs

    @torch.jit.export
    def infer(self, x, input_lengths, spk_embeddings):
        device = x.device
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x.to(device))), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        spk_embeddings_trans = self.spk_embedding_tf_layer(
            spk_embeddings).unsqueeze(1).expand(-1, outputs.size(1), -1)

        final_outputs = torch.cat([outputs, spk_embeddings_trans], dim=-1)

        return final_outputs


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.attention_mode = hparams.attention_mode

        self.prenet = Prenet(
            n_mel_channels,
            [prenet_dim, prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)

        if self.attention_mode == "default":
            self.attention_layer = Attention(
                attention_rnn_dim, encoder_embedding_dim,
                attention_dim, attention_location_n_filters,
                attention_location_kernel_size)

        elif self.attention_mode == "GMM":
            self.kernel = hparams.gmm_kernel
            self.attention_layer = GMMAttention(attention_rnn_dim, attention_dim,
                                                hparams.gmm_kernel, hparams.delta_bias, hparams.sigma_bias)

        elif self.attention_mode == "FA":
            self.attention_layer = ForwardAttentionV2(
                attention_rnn_dim, encoder_embedding_dim,
                attention_dim, attention_location_n_filters,
                attention_location_kernel_size)

        elif self.attention_mode == "SMA":
            self.attention_layer = StepwiseMonotonicAttention(
                attention_rnn_dim, encoder_embedding_dim, attention_dim)

        elif self.attention_mode == "DCA":
            self.attention_layer = DynamicConvolutionAttention(
                attention_rnn_dim, encoder_embedding_dim, attention_dim)

        elif self.attention_mode == "GMM_Evo":
            self.attention_layer = GMMAttentionEvo(
                attention_rnn_dim, encoder_embedding_dim, attention_dim)

        else:
            raise ValueError("unsupported attention mode")

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step)

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, n_frames_per_step,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mel_channels,
            dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device)

        attention_weights_cum = None
        processed_memory = None

        if self.attention_mode == "default":
            attention_weights_cum = torch.zeros(
                B, MAX_TIME, dtype=dtype, device=device)
            processed_memory = self.attention_layer.memory_layer(memory)

        elif self.attention_mode == "GMM":
            self.mu = memory.new_zeros(B, self.kernel, 1)  # [B, K, 1]
            self.t = torch.arange(
                MAX_TIME, device=memory.device, dtype=torch.float)
            self.t = self.t.expand(B, self.kernel, MAX_TIME)

        elif self.attention_mode == "FA":
            attention_weights_cum = memory.new_zeros(B, MAX_TIME)
            self.log_alpha = memory.new_zeros(B, MAX_TIME).fill_(-float(1e20))
            self.log_alpha[:, 0].fill_(0.)
            processed_memory = self.attention_layer.memory_layer(memory)

        elif self.attention_mode == "SMA":
            self.attention_layer.init_attention(memory)

        elif self.attention_mode == "DCA":
            self.attention_layer.init_attention(memory)

        elif self.attention_mode == "GMM_Evo":
            self.attention_layer.init_attention(memory)

        else:
            raise ValueError("unsupported attention mode")

        return (attention_hidden, attention_cell, decoder_hidden,
                decoder_cell, attention_weights, attention_weights_cum,
                attention_context, processed_memory)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        '''
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        '''
        decoder_inputs = decoder_inputs.permute(2, 0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, mel_lengths=None):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B, n_frames_per_step) -> (B, T_out, n_frames_per_step)
        gate_outputs = gate_outputs.transpose(0, 1)
        # (B, T_out, n_frames_per_step) -> (B, T_out)
        gate_outputs = gate_outputs.contiguous().view(gate_outputs.size(0), -1)
        # (T_out, B, n_mel_channels * n_frames_per_step) -> (B, T_out, n_mel_channels * n_frames_per_step)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        if mel_lengths is not None:
            mel_lengths *= self.n_frames_per_step

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_hidden, attention_cell,
               decoder_hidden, decoder_cell, attention_weights,
               attention_weights_cum, attention_context, memory,
               processed_memory, mask):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training)

        if self.attention_mode == "default":
            attention_weights_cat = torch.cat(
                (attention_weights.unsqueeze(1),
                 attention_weights_cum.unsqueeze(1)), dim=1)
            attention_context, attention_weights = self.attention_layer(
                attention_hidden, memory, processed_memory,
                attention_weights_cat, mask)

            attention_weights_cum += attention_weights

        elif self.attention_mode == "GMM":
            attention_context, attention_weights, self.mu = self.attention_layer(
                attention_hidden, memory, self.mu, self.t, mask)

        elif self.attention_mode == "FA":
            attention_weights_cat = torch.cat(
                (attention_weights.unsqueeze(1),
                 attention_weights_cum.unsqueeze(1)), dim=1)
            attention_context, attention_weights, self.log_alpha = self.attention_layer(
                attention_hidden, memory, processed_memory, attention_weights_cat, mask, self.log_alpha)
            attention_weights_cum += attention_weights

        elif self.attention_mode == "SMA":
            attention_weights = self.attention_layer(attention_hidden, memory)
            attention_context = torch.bmm(
                attention_weights.unsqueeze(1), memory)
            attention_context = attention_context.squeeze(1)

        elif self.attention_mode == "DCA":
            attention_weights = self.attention_layer(attention_hidden, memory)
            attention_context = torch.bmm(
                attention_weights.unsqueeze(1), memory)
            attention_context = attention_context.squeeze(1)

        elif self.attention_mode == "GMM_Evo":
            attention_weights = self.attention_layer(attention_hidden, memory)
            attention_context = torch.bmm(
                attention_weights.unsqueeze(1), memory)
            attention_context = attention_context.squeeze(1)

        else:
            raise ValueError("not supported attention mode")

        decoder_input = torch.cat(
            (attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1)
        # [B, n_mel_channels * n_frames_per_step]
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # [B, n_frames_per_step]
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context)

    @torch.jit.ignore
    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments))

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device)
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device)

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = torch.le(torch.sigmoid(gate_output),
                           self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished*dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, mel_lengths)

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_rnn_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, decoder_no_early_stopping):
        super(Tacotron2, self).__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(encoder_n_convolutions,
                               symbols_embedding_dim,
                               encoder_kernel_size)
        self.decoder = Decoder(n_mel_channels, n_frames_per_step,
                               encoder_embedding_dim + hparams.spk_embedding_dim, attention_dim,
                               attention_location_n_filters,
                               attention_location_kernel_size,
                               attention_rnn_dim, decoder_rnn_dim,
                               prenet_dim, max_decoder_steps,
                               gate_threshold, p_attention_dropout,
                               p_decoder_dropout,
                               not decoder_no_early_stopping)
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim,
                               postnet_kernel_size,
                               postnet_n_convolutions)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths):
        # type: (List[Tensor], Tensor) -> List[Tensor]
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, output_lengths, spk_embeddings = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(
            embedded_inputs, input_lengths, spk_embeddings)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def infer(self, inputs, input_lengths, spk_embeddings):

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(
            embedded_inputs, input_lengths, spk_embeddings)
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments
