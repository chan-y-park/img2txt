import heapq
import numpy as np

class Beam:
    def __init__(self, seq, score, prev_state):
        self.seq = seq
        self.score = score
        self.prev_state = prev_state

    def __lt__(self, other):
        # TODO: If scores are the same, compare lengths.
        assert isinstance(other, Beam)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Beam)
        return self.score == other.score


class BeamMinHeap:
    def __init__(self, beam_size):
        self.beam_size = beam_size
        self._data = []

    def size(self):
        return len(self._data)

    def push(self, item):
        if len(self._data) >= self.beam_size:
            heapq.heappushpop(self._data, item)
        else:
            heapq.heappush(self._data, item)

    def pop_all(self):
        data = self._data
        self._data = []
        return data
        

def run_inference_using_beam_search(
    model,
    images_array,
):
    beam_size = model._beam_size
    max_sequence_length = model._config['max_sequence_length']

    # Feed images, fetch RNN initial states.
    feed_dict = {
        model._tf_graph.get_tensor_by_name(
            'inference_input_images:0'
        ): images_array,
    }
    fetch_dict = {
        'rnn/inference_initial_states':
            model._tf_graph.get_tensor_by_name(
                'rnn/inference_initial_states:0'
            ),
        'convnet/inference_predictions':
            model._tf_graph.get_tensor_by_name(
                'convnet/inference_predictions:0'
            ),
        # TODO: Check the latency of fetching the following tensor.
        'rnn/word_embedding':
            model._tf_graph.get_tensor_by_name(
                'rnn/word_embedding:0'
            ),
        'image_embedding/inference_image_embeddings':
            model._tf_graph.get_tensor_by_name(
                'image_embedding/inference_image_embeddings:0'
            ),
    }
    rd_init = model._tf_session.run(
        fetches=fetch_dict,
        feed_dict=feed_dict,
    )

    prev_rnn_states = rd_init['rnn/inference_initial_states']
    prev_state = prev_rnn_states[0]
    state_size = len(prev_state)
    assert (len(prev_rnn_states) == beam_size)
    assert (
        np.all(
            (prev_rnn_states - prev_state) == 0
        )
    )

#    inputs = np.array(
#        [[model._vocabulary.start_word_id] for i in range(minibatch_size)]
#    )
#    output_seqs = np.zeros(
#        shape=(beam_size, max_sequence_length),
#        dtype=np.int32,
#    )

    complete_beams = BeamMinHeap(beam_size)
    partial_beams = BeamMinHeap(beam_size)
    partial_beams.push(
        Beam(
            seq=[model._vocabulary.start_word_id],
            score=0,
            prev_state=prev_state,
        )
    )

    # For max_sequence_length, feed input seqs
    # and fetch word probabilities & new RNN states.
    for t in range(max_sequence_length):
        beams = partial_beams.pop_all()
        inputs = np.empty(
            shape=(model._beam_size, 1),
            dtype=np.int32,
        )
        prev_rnn_states = np.empty(
            shape=(model._beam_size, state_size),
            dtype=np.float32,
        )
        for i_b, a_beam in enumerate(beams):
            inputs[i_b] = a_beam.seq[-1]
            prev_rnn_states[i_b] = a_beam.prev_state

        feed_dict = {
            model._tf_graph.get_tensor_by_name(
                'inference_inputs:0'
            ): inputs,
            model._tf_graph.get_tensor_by_name(
                'rnn/inference_prev_states:0'
            ): prev_rnn_states,
        }
        fetch_dict = {
            'rnn/inference_new_states': model._tf_graph.get_tensor_by_name(
                'rnn/inference_new_states:0'
            ),
            'rnn/fc/inference_word_logits': (
                model._tf_graph.get_tensor_by_name(
                    'rnn/fc/inference_predictions:0'
                )
            ),
            'rnn/fc/inference_word_ids': model._tf_graph.get_tensor_by_name(
                'rnn/fc/inference_predictions:1'
            ),
        }
        rd = model._tf_session.run(
            fetches=fetch_dict,
            feed_dict=feed_dict,
        )

        for i_b, a_beam in enumerate(beams):
            prev_state = rd['rnn/inference_new_states'][i_b]
            word_logits = rd['rnn/fc/inference_word_logits'][i_b]
            word_ids = rd['rnn/fc/inference_word_ids'][i_b]
            assert (len(prev_state) == state_size)
            assert (len(word_logits) == beam_size)
            assert (len(word_ids) == beam_size)

            for i_w in range(beam_size):
                word_logit = word_logits[i_w]
                word_id = word_ids[i_w]

                a_beam.seq += [word_id]
                a_beam.score += word_logit
                a_beam.prev_state = prev_state

                if (word_id == model._vocabulary.end_word_id):
                    complete_beams.push(a_beam)
                else:
                    partial_beams.push(a_beam)
            
            import pdb
            pdb.set_trace()

        if (partial_beams.size() == 0):
            break

    if (complete_beams.size() == 0):
        complete_beams = partial_beams
        
    beams = complete_beams.pop_all()
    # TODO: Sort seqs by scores and return all the seqs.
    scores = [beam.score for beam in beams]
    i_best = np.argmax(scores)
    rd = {
        'output_sequences': [beams[i_best].seq],
    }
    for var_name in [
        'rnn/word_embedding',
        'image_embedding/inference_image_embeddings',
        'convnet/inference_predictions',
    ]:
        rd[var_name] = rd_init[var_name]
    return rd


