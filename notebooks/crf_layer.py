"""
Self-contained CRF layer for TensorFlow 2 / Keras.

Implements the standard linear-chain CRF forward algorithm (log-partition
function via log-sum-exp) and Viterbi decoding, using only core TensorFlow
ops. Written from scratch to avoid depending on `tensorflow-addons`, which
is deprecated and causes dependency-resolution conflicts on recent
TensorFlow/Python versions.
"""

import tensorflow as tf


def crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params):
    """
    Computes the unnormalized score of a tag sequence.
    inputs: (batch, T, num_tags) emission scores
    tag_indices: (batch, T) int32 gold tags
    sequence_lengths: (batch,) int32 actual (unpadded) lengths
    transition_params: (num_tags, num_tags)
    Returns: (batch,) score per example
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    # Unary (emission) score: sum of inputs[b, t, tag_indices[b, t]] for t < seq_len
    mask = tf.sequence_mask(sequence_lengths, maxlen=max_len, dtype=inputs.dtype)  # (batch, T)
    batch_idx = tf.tile(tf.range(batch_size)[:, None], [1, max_len])
    time_idx = tf.tile(tf.range(max_len)[None, :], [batch_size, 1])
    gather_idx = tf.stack([batch_idx, time_idx, tag_indices], axis=-1)  # (batch, T, 3)
    unary_scores = tf.gather_nd(inputs, gather_idx)  # (batch, T)
    unary_scores = tf.reduce_sum(unary_scores * mask, axis=1)  # (batch,)

    # Binary (transition) score: sum of transition_params[tag_t, tag_{t+1}] for valid consecutive pairs
    tag_pairs_from = tag_indices[:, :-1]  # (batch, T-1)
    tag_pairs_to = tag_indices[:, 1:]     # (batch, T-1)
    trans_gather_idx = tf.stack([tag_pairs_from, tag_pairs_to], axis=-1)  # (batch, T-1, 2)
    binary_scores = tf.gather_nd(transition_params, trans_gather_idx)  # (batch, T-1)
    # a transition t->t+1 is valid only if t+1 < seq_len (i.e. both tags are real, not padding)
    trans_mask = mask[:, 1:]  # transition into position t+1 is valid iff position t+1 is real
    binary_scores = tf.reduce_sum(binary_scores * trans_mask, axis=1)  # (batch,)

    return unary_scores + binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """
    Computes the log partition function (normalizer) Z via the forward
    algorithm in log-space (log-sum-exp), respecting variable sequence
    lengths using masking.
    Returns: (batch,) log Z per example
    """
    max_len = tf.shape(inputs)[1]
    mask = tf.sequence_mask(sequence_lengths, maxlen=max_len, dtype=inputs.dtype)  # (batch, T)

    inputs_t = tf.transpose(inputs, [1, 0, 2])       # (T, batch, num_tags)
    mask_t = tf.transpose(mask, [1, 0])              # (T, batch)

    def step(alpha_prev, elems):
        inp_t, mask_slice = elems
        # alpha_prev: (batch, num_tags)
        broadcast = alpha_prev[:, :, None] + transition_params[None, :, :]  # (batch, from, to)
        new_alpha = tf.reduce_logsumexp(broadcast, axis=1) + inp_t          # (batch, num_tags)
        mask_slice = mask_slice[:, None]  # (batch, 1)
        alpha = mask_slice * new_alpha + (1.0 - mask_slice) * alpha_prev
        return alpha

    alpha_0 = inputs_t[0]  # (batch, num_tags)
    alphas = tf.scan(step, (inputs_t[1:], mask_t[1:]), initializer=alpha_0)  # (T-1, batch, num_tags)
    all_alphas = tf.concat([alpha_0[None, :, :], alphas], axis=0)  # (T, batch, num_tags)

    # gather the alpha at each example's true final timestep (seq_len - 1)
    final_idx = tf.maximum(sequence_lengths - 1, 0)  # (batch,)
    batch_size = tf.shape(inputs)[0]
    gather_idx = tf.stack([final_idx, tf.range(batch_size)], axis=-1)  # (batch, 2) into (T, batch, ...)
    final_alpha = tf.gather_nd(all_alphas, gather_idx)  # (batch, num_tags)

    return tf.reduce_logsumexp(final_alpha, axis=-1)  # (batch,)


def crf_log_likelihood(inputs, tag_indices, sequence_lengths, transition_params):
    """Returns per-example log-likelihood (batch,)."""
    seq_score = crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)
    log_z = crf_log_norm(inputs, sequence_lengths, transition_params)
    return seq_score - log_z


def crf_decode(inputs, transition_params, sequence_lengths):
    """
    Viterbi decoding.
    Returns: (batch, T) int32 best tag sequence (padded positions arbitrary,
    caller should mask using sequence_lengths).
    """
    max_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]
    mask = tf.sequence_mask(sequence_lengths, maxlen=max_len, dtype=inputs.dtype)

    inputs_t = tf.transpose(inputs, [1, 0, 2])  # (T, batch, num_tags)
    mask_t = tf.transpose(mask, [1, 0])         # (T, batch)

    def step(prev, elems):
        prev_score, _ = prev
        inp_t, mask_slice = elems
        broadcast = prev_score[:, :, None] + transition_params[None, :, :]  # (batch, from, to)
        best_prev_tag = tf.argmax(broadcast, axis=1, output_type=tf.int32)   # (batch, to)
        best_score = tf.reduce_max(broadcast, axis=1) + inp_t                # (batch, to)
        mask_slice_exp = mask_slice[:, None]
        new_score = mask_slice_exp * best_score + (1.0 - mask_slice_exp) * prev_score
        return new_score, best_prev_tag

    score_0 = inputs_t[0]  # (batch, num_tags)
    backptr_0 = tf.zeros_like(score_0, dtype=tf.int32)

    scores, backptrs = tf.scan(
        step, (inputs_t[1:], mask_t[1:]), initializer=(score_0, backptr_0)
    )
    # scores: (T-1, batch, num_tags), backptrs: (T-1, batch, num_tags)

    all_scores = tf.concat([score_0[None, :, :], scores], axis=0)  # (T, batch, num_tags)

    batch_size = tf.shape(inputs)[0]
    final_idx = tf.maximum(sequence_lengths - 1, 0)
    gather_idx = tf.stack([final_idx, tf.range(batch_size)], axis=-1)
    final_scores = tf.gather_nd(all_scores, gather_idx)  # (batch, num_tags)
    best_last_tag = tf.argmax(final_scores, axis=-1, output_type=tf.int32)  # (batch,)

    # Backtrack. backptrs has T-1 steps (for t=1..T-1); we walk backwards.
    def backtrack_step(next_tag, backptr_t):
        # backptr_t: (batch, num_tags) — best_prev_tag when arriving at each `to` tag at this step
        batch_idx = tf.range(tf.shape(next_tag)[0])
        idx = tf.stack([batch_idx, next_tag], axis=-1)
        return tf.gather_nd(backptr_t, idx)

    tags_reversed = tf.scan(
        backtrack_step,
        tf.reverse(backptrs, axis=[0]),
        initializer=best_last_tag,
    )  # (T-1, batch) — these are the tags for positions T-2 downto 0, in reverse order

    tags_reversed = tf.reverse(tags_reversed, axis=[0])  # now in forward order, positions 0..T-2
    all_tags = tf.concat([tags_reversed, best_last_tag[None, :]], axis=0)  # (T, batch)
    all_tags = tf.transpose(all_tags, [1, 0])  # (batch, T)

    return all_tags


class CRF(tf.keras.layers.Layer):
    """Keras layer wrapping the CRF transition matrix as a trainable weight."""

    def __init__(self, num_tags, **kwargs):
        super().__init__(**kwargs)
        self.num_tags = num_tags

    def build(self, input_shape):
        self.transitions = self.add_weight(
            name="transitions",
            shape=(self.num_tags, self.num_tags),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, potentials, sequence_lengths):
        """Returns decoded tag sequence (for inference)."""
        return crf_decode(potentials, self.transitions, sequence_lengths)

    def loss(self, potentials, tag_indices, sequence_lengths):
        """Returns scalar negative log-likelihood loss."""
        log_likelihood = crf_log_likelihood(potentials, tag_indices, sequence_lengths, self.transitions)
        return -tf.reduce_mean(log_likelihood)
