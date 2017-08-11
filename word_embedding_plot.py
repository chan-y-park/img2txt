import numpy as np
import sklearn

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, LabelSet, Label
from bokeh.embed import components

from dataset import Vocabulary


def get_sentence_vector(sequence, word_embedding):
    sentence_vector = 0
    for word_id in sequence:
        word_vec = word_embedding[word_id]
        sentence_vector += word_vec
    return sentence_vector


def get_nearby_word_vectors(
    vectors, word_embedding, num_nearby_word_vectors,
    metric='cosine',
):
    # vectors: [batch_size, embedding_size]
    # word_embedding: [vocabulary_size, embedding_size]
    batch_size, embedding_size = vectors.shape
    vocabulary_size, _ = word_embedding.shape
    if metric == 'cosine':
        distance_fn = sklearn.metrics.pairwise.cosine_distances
    elif metric == 'euclidean':
        distance_fn = sklearn.metrics.pairwise.euclidean_distances
    else:
        raise ValueError('Unknown metric: {}'.format(metric))

    # distances: [batch_size, vocabulary_size]
    distances = distance_fn(
        vectors,
        word_embedding,
    )
    # nearby_word_ids = [batch_size, num_nearby_word_vectors]
    nearby_word_ids = np.argsort(
        distances,
        axis=1,
    )[:, 1:num_nearby_word_vectors + 1]     # exclude the vector itself.
    flattened_nearby_word_ids = nearby_word_ids.reshape(
        (batch_size * num_nearby_word_vectors)
    )
    # truncated_distances: [batch_size, num_nearby_word_vectors]
    truncated_distances = (
        distances.reshape((-1))[flattened_nearby_word_ids]
        .reshape((batch_size, num_nearby_word_vectors))
    )
    return  (
        truncated_distances,
        nearby_word_ids,
        word_embedding[flattened_nearby_word_ids,:].reshape(
            (batch_size, num_nearby_word_vectors, embedding_size)
        ),
    )


def get_cosine_similarity(vector_1, vector_2):
    return (
        np.sum(vector_1 * vector_2)
        / np.linalg.norm(vector_1)
        / np.linalg.norm(vector_2)
    )


def get_cosine_distance(vector_1, vector_2):
    return (1.0 - get_cosine_similarity(vector_1, vector_2)) 


def get_euclidean_distance(vector_1, vector_2):
    return np.linalg.norm(vector_1 - vector_2) 


def get_word_embedding_plot_of_nearby_words(
    image_vector,
    sequence,
    word_embedding,
    vocabulary,
    metric='cosine',
    num_nearby_words=5,
    use_pca=False,
    plot_width=600,
    plot_height=600,
):
    if metric == 'cosine':
        tsne_metric=sklearn.metrics.pairwise.cosine_distances
        get_distance = get_cosine_distance
    elif metric == 'euclidean':
        tsne_metric=sklearn.metrics.pairwise.euclidean_distances
        get_distance = get_euclidean_distance
    else:
        raise ValueError('Unknown metric: {}'.format(metric))

    vocabulary_size, embedding_size = word_embedding.shape

    sentence_vector = get_sentence_vector(sequence, word_embedding)
    vectors = np.concatenate(
        (image_vector[np.newaxis,:],
         sentence_vector[np.newaxis,:],
         word_embedding[sequence,:])
    )
    distances, nearby_word_ids, nearby_word_vectors = get_nearby_word_vectors(
        vectors,
        word_embedding,
        num_nearby_words,
        metric=metric,
    )
#    words = [vocabulary.get_word_of_id(word_id) for word_id in word_ids]
#    sentence = vocabulary.get_sentence_from_word_ids(sequence) 

    # NOTE: concatenated_vectors = [
    #   image_vector,
    #   sentence_vector,
    #   word_in_sentence_vectors,
    #   flattened_nearby_word_vectors,
    # ]
    concatenated_vectors = np.concatenate(
        (vectors, nearby_word_vectors.reshape((-1, embedding_size))),
    )
#    pca = sklearn.decomposition.PCA(n_components=50)
#    concatenated_vectors = pca.fit_transform(concatenated_vectors)
    tsne = TSNE(metric=tsne_metric)
    transformed_vectors = tsne.fit_transform(concatenated_vectors)
    image_vector_distance = get_distance(
        image_vector,
        sentence_vector,
    )

    hover = HoverTool(
        tooltips=[
            ('distance_from', '@source'),
            ('distance', '@distance'),
        ]
    )
    bokeh_figure = figure(
        tools='reset,box_zoom,pan,wheel_zoom,save,tap',
        plot_width=plot_width,
        plot_height=plot_height,
        title='Word Embedding',
#        x_range=,
#        y_range=,
    )
    bokeh_figure.add_tools(hover)

#    bokeh_figure.grid.grid_line_color = None
#    bokeh_figure.xaxis.major_tick_line_color = None
#    bokeh_figure.xaxis.minor_tick_line_color = None
#    bokeh_figure.yaxis.major_tick_line_color = None
#    bokeh_figure.yaxis.minor_tick_line_color = None
#    bokeh_figure.xaxis.major_label_text_font_size = '0pt'
#    bokeh_figure.yaxis.major_label_text_font_size = '0pt'

#    source = ColumnDataSource(
#        data={
#            'x': vectors[:, 0],
#            'y': vectors[:, 1],
#            'color': ['red', 'black'] + ['blue'] * num_word_vectors,
#            'text': ['IMAGE', sentence] + words,
#            'distance': [image_vector_distance, 0] + list(distances),
#            'alpha': [1.0, 1.0] + [0.5] * num_word_vectors,
#        }
#    )
    num_source_vectors = 2 + len(sequence)
    tsne_image_vector = transformed_vectors[0]
    tsne_sentence_vector = transformed_vectors[1]
    tsne_sentence_word_vectors = transformed_vectors[2: num_source_vectors]
    tsne_nearby_word_vectors = (
        transformed_vectors[num_source_vectors:]
        .reshape(
            (2 + len(sequence), num_nearby_words, -1)
        )
    )

    sentence_words = [vocabulary.get_word_of_id(word_id)
                      for word_id in sequence]
    source_words = ['IMAGE', 'SENTENCE'] + sentence_words
    data_dict = {
        'x': transformed_vectors[:num_source_vectors, 0],
        'y': transformed_vectors[:num_source_vectors, 1],
        'color': ['navy'] * num_source_vectors,
        'source': source_words,
        'text': source_words,
        'distance': [0] * num_source_vectors,
    }
    cds = ColumnDataSource(data_dict)
    bokeh_figure.circle_cross(
        x='x', y='y',
        size=5,
        color='color',
        source=cds,
    )
    labels = LabelSet(
        x='x', y='y', text='text', level='glyph',
        x_offset=5, y_offset=5, source=cds, render_mode='canvas',
    )
    bokeh_figure.add_layout(labels)

    nearby_words = [
        [vocabulary.get_word_of_id(word_id) for word_id in word_ids]
        for word_ids in nearby_word_ids
    ]
    for i_src_vec in range(num_source_vectors):
        start = (i_src_vec + 1)* num_nearby_words
        stop = start + num_nearby_words
        target_vectors = transformed_vectors[start:stop,:]
        data_dict = {
            'x': target_vectors[:, 0],
            'y': target_vectors[:, 1],
            'color': ['olive'] * num_nearby_words,
            'source': [source_words[i_src_vec]] * num_nearby_words,
            'text': nearby_words[i_src_vec],
            'distance': distances[i_src_vec],
        }
        cds = ColumnDataSource(data_dict)
        bokeh_figure.circle(
            x='x', y='y',
            size=5,
            color='color',
            source=cds,
        )
        labels = LabelSet(
            x='x', y='y', text='text', level='glyph',
            x_offset=5, y_offset=5, source=cds, render_mode='canvas',
        )
        bokeh_figure.add_layout(labels)

    return bokeh_figure


def get_word_embedding_plot_of_sentence(
    sequence,
    word_embedding,
    vocabulary_file_path,
    metric='cosine',
    num_nearby_words=40,
    plot_width=600,
    plot_height=600,
):
    vocabulary = Vocabulary(file_path=vocabulary_file_path)
    if metric == 'cosine':
        tsne_metric=sklearn.metrics.pairwise.cosine_distances
        get_distance = get_cosine_distance
    elif metric == 'euclidean':
        tsne_metric=sklearn.metrics.pairwise.euclidean_distances
        get_distance = get_euclidean_distance
    else:
        raise ValueError('Unknown metric: {}'.format(metric))

    vocabulary_size, embedding_size = word_embedding.shape

    sentence_vector = get_sentence_vector(sequence, word_embedding)
    distances, nearby_word_ids, nearby_word_vectors = get_nearby_word_vectors(
        sentence_vector[np.newaxis,:],
        word_embedding,
        num_nearby_words,
        metric=metric,
    )

    concatenated_vectors = np.concatenate(
        (sentence_vector[np.newaxis,:],
         nearby_word_vectors.reshape((-1, embedding_size))),
    )
    tsne = TSNE(metric=tsne_metric)
    transformed_vectors = tsne.fit_transform(concatenated_vectors)

    bokeh_figure = figure(
        tools='reset,box_zoom,pan,wheel_zoom,save,tap',
        plot_width=plot_width,
        plot_height=plot_height,
        title='Word Embedding',
    )

    tsne_sentence_vector = transformed_vectors[0]
    tsne_nearby_word_vectors = transformed_vectors[1:]

    sentence_words = [vocabulary.get_word_of_id(word_id)
                      for word_id in sequence]
    data_dict = {
        'x': [tsne_sentence_vector[0]],
        'y': [tsne_sentence_vector[1]],
        'color': ['navy'],
        'text': ['SENTENCE'],
        'distance': [0],
    }
    cds = ColumnDataSource(data_dict)
    bokeh_figure.circle_cross(
        x='x', y='y',
        size=10,
        color='color',
        source=cds,
    )
    labels = LabelSet(
        x='x', y='y', text='text', level='glyph',
        x_offset=5, y_offset=5, source=cds, render_mode='canvas',
    )
    bokeh_figure.add_layout(labels)

    nearby_words = [
        vocabulary.get_word_of_id(word_id)
        for word_id in nearby_word_ids[0]
    ]
    data_dict = {
        'x': tsne_nearby_word_vectors[:, 0],
        'y': tsne_nearby_word_vectors[:, 1],
        'color': ['olive'] * num_nearby_words,
        'text': nearby_words,
        'distance': distances[0],
    }
    cds = ColumnDataSource(data_dict)
    bokeh_figure.circle(
        x='x', y='y',
        size=5,
        color='color',
        source=cds,
    )
    labels = LabelSet(
        x='x', y='y', text='text', level='glyph',
        x_offset=5, y_offset=5, source=cds, render_mode='canvas',
    )
    bokeh_figure.add_layout(labels)

    return bokeh_figure


def get_tsne_of_word_embedding(
    word_embedding,
    save_file_path=None,
    tsne_init='pca',
    tsne_metric='cosine',
    tsne_verbosity=2,
):
    tsne = TSNE(
        init=tsne_init,
        metric=tsne_metric,
        verbose=tsne_verbosity,
    )
    tsne_word_vectors = tsne.fit_transform(word_embedding)
    if save_file_path is not None:
        np.save(save_file_path, tsne_word_vectors)

    return tsne_word_vectors


def load_tsne_of_word_embedding(
    file_path,
):
    return np.load(file_path)


def get_word_embedding_plot(
    sequence,
    tsne_file_path,
    vocabulary_file_path=None,
    vocabulary=None,
    notebook=False,
    plot_width=600,
    plot_height=600,
):
    if vocabulary_file_path is not None:
        vocabulary = Vocabulary(file_path=vocabulary_file_path)
    elif vocabulary is None:
        raise ValueError(
            'Either vocabulary_file_path or vocabulary '
            'should be provided.'
        )
    if tsne_file_path is None:
        raise ValueError

    sequence = np.unique(sequence)
    hover = HoverTool(
        tooltips=[
            ('word', '@word'),
        ]
    )
    bokeh_figure = figure(
        tools='reset,box_zoom,pan,wheel_zoom,save,tap',
        plot_width=plot_width,
        plot_height=plot_height,
        title=(
            'Click on legend entries to hide the corresponding data.'
        ),
    )
    bokeh_figure.add_tools(hover)

    word_embedding = load_tsne_of_word_embedding(
        tsne_file_path
    )

    sentence_words = [vocabulary.get_word_of_id(word_id)
                      for word_id in sequence]
    sentence_words_data_dict = {
        'x': word_embedding[sequence, 0],
        'y': word_embedding[sequence, 1],
        'color': ['navy'] * len(sequence),
        'word': sentence_words,
    }
    sentence_words_data_source = ColumnDataSource(sentence_words_data_dict)
    bokeh_figure.circle_cross(
        x='x', y='y',
        size=10,
        color='color',
        fill_alpha=0.5,
        muted_alpha=0.2,
        legend='sentence words',
        source=sentence_words_data_source,
    )
    labels = LabelSet(
        x='x', y='y', text='word', level='glyph',
        x_offset=5, y_offset=5, render_mode='canvas',
        source=sentence_words_data_source,
    )
    bokeh_figure.add_layout(labels)

    other_word_ids = np.delete(
        np.array(range(vocabulary.get_size())),
        sequence,
    )
    other_words = [vocabulary.get_word_of_id(word_id)
                   for word_id in other_word_ids]
    other_words_data_dict = {
        'x': word_embedding[other_word_ids, 0],
        'y': word_embedding[other_word_ids, 1],
        'word': other_words,
    }
    other_words_data_source = ColumnDataSource(other_words_data_dict)
    bokeh_figure.circle(
        x='x', y='y',
        size=10,
        color='gray',
        fill_alpha=0.1,
        line_alpha=0.1,
        muted_alpha=0.05,
        muted_color='gray',
        legend='other words',
        source=other_words_data_source,
    )
    
    bokeh_figure.legend.location = 'top_left'
    bokeh_figure.legend.click_policy = 'mute'

    if notebook:
        return bokeh_figure
    else:
        script, div = components(bokeh_figure)
        return {
            'script': script,
            'div': div,
        }
