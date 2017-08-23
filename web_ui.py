import os
import uuid
import time
import glob
import random

import numpy as np
import flask

#from werkzeug.utils import secure_filename
from PIL import Image

from model import Image2Text
from convnet import resize_image, preprocess_image
from word_embedding_plot import get_word_embedding_plot

# TODO: display a message to the user browser.
message = print

ALLOWED_EXTENTIONS = ['jpg', 'jpeg']
UPLOADED_IMAGES_DIR = 'uploaded_images'
INPUT_IMAGES_DIR = 'input_images'


#def allowed_file(filename):
#    if '.' in filename:
#        if filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS:
#            return True
#    return False


def get_image_path(directory, image_id, ext='jpg'):
    return os.path.join(
        directory,
        '.'.join([image_id, ext]),
    )


class Img2TxtWebApp(flask.Flask):
    def __init__(self):
        super().__init__('Img2Txt')
        for directory in [UPLOADED_IMAGES_DIR, INPUT_IMAGES_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        self._img2txt_inference = Image2Text(
            vocabulary_file_path='inference/vocabulary.json',
            config_file_path='inference/config.json',
            checkpoint_save_path='inference/img2txt',
            inference_only=True,
            minibatch_size=1,
        )
    def get_inference_result(self, image_id):
        uploaded_image_path = get_image_path(
            UPLOADED_IMAGES_DIR,
            image_id,
        )
        uploaded_image = Image.open(uploaded_image_path)

        cfg_convnet = self._img2txt_inference._config['convnet']
        input_image_size = cfg_convnet['input_image_shape'][0]
        input_image = resize_image(uploaded_image, input_image_size)
        input_image.save(get_image_path(INPUT_IMAGES_DIR, image_id))

        preprocessed_image = preprocess_image(
            convnet_name=cfg_convnet['name'],
            image=input_image,
        )
        rd = self._img2txt_inference.get_inference(
            preprocessed_image[np.newaxis,:],
        )

        vocabulary = self._img2txt_inference._vocabulary 
        word_embedding = rd['rnn/word_embedding']
        sequence = rd['output_sequences'][0]
        post_seq = vocabulary.get_postprocessed_sequence(sequence)

        bokeh_plot = get_word_embedding_plot(
            sequence=post_seq,
            tsne_file_path='ms_coco_word_embedding_pca_cosine.npy',
            vocabulary=vocabulary,
        )

        return {
            'image_id': image_id,
            'caption': vocabulary.get_sentence_from_word_ids(post_seq),
            'bokeh_plot_script': bokeh_plot['script'],
            'bokeh_plot_div': bokeh_plot['div'],
        }

def index():
    return flask.redirect(flask.url_for('config'))


def get_random_image(
    image_dir='test_images',
):
    image_file_path = random.choice(glob.glob("test_images/*.jpg"))
    return Image.open(image_file_path)


def config():
    app = flask.current_app
    image_file = None
    if flask.request.method == 'POST':
        image_id = str(uuid.uuid4())
        image_file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            '{}.jpg'.format(image_id)
        )
        if 'use_random_image' in flask.request.form:
            if eval(flask.request.form['use_random_image']):
                image = get_random_image()
                image.save(image_file_path)
        else:
            if 'image_file' not in flask.request.files:
                print('No image file in the html request form.')
                return flask.redirect(flask.request.url)

            image_file = flask.request.files['image_file']
            if image_file.filename == '':
                return flask.render_template(
                    'config.html',
                    error_message='No selected file.',
                )

            if image_file:
                try:
                    pil_image = Image.open(image_file)
                    pil_image.save(image_file_path)
                except IOError:
                    error_message='The image cannot be opened and identified.'
                    return flask.render_template(
                        'config.html',
                        error_message=error_message,
                    )

        return flask.render_template(
            'config.html',
            image_id=image_id,
        )
    else:
        # TODO: Load a default image file.
        return flask.render_template('config.html')


def get_image(directory, image_id):
    rv = flask.send_file(
        get_image_path(directory, image_id),
        mimetype='image/jpg',
        cache_timeout=0,
        as_attachment=False,
        attachment_filename='image.jpg',
    )
    rv.set_etag(str(time.time()))
    return rv


def uploaded_image(image_id):
    return get_image(UPLOADED_IMAGES_DIR, image_id)


def input_image(image_id):
    return get_image(INPUT_IMAGES_DIR, image_id)


def show_results():
    app = flask.current_app
    image_id = flask.request.form['image_id']

    rd_inference = app.get_inference_result(image_id)

    return flask.render_template(
        'result.html',
        **rd_inference,
    )


def get_web_app(
    debug=False,
    upload_folder='uploaded_images',
):
    web_app = Img2TxtWebApp()
    web_app.config.update(
        DEBUG=debug,
        UPLOAD_FOLDER=upload_folder,
        SECRET_KEY='img2txt secret key',
    )

    web_app.add_url_rule(
        '/',
        'index',
        index,
        methods=['GET'],
    )
    web_app.add_url_rule(
        '/config',
        'config',
        config,
        methods=['GET', 'POST'],
    )
    web_app.add_url_rule(
        '/input_image/<image_id>',
        'input_image',
        input_image,
        methods=['GET'],
    )
    web_app.add_url_rule(
        '/uploaded_image/<image_id>',
        'uploaded_image',
        uploaded_image,
        methods=['GET'],
    )
    web_app.add_url_rule(
        '/results',
        'show_results',
        show_results,
        methods=['POST'],
    )

    return web_app
