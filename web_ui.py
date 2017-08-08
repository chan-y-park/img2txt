import flask

from werkzeug.utils import secure_filename
from PIL import Image

from model import Image2Text
from convnet import preprocess_image

# TODO: display a message to the user browser.
message = print

ALLOWED_EXTENTIONS = ['jpg', 'jpeg']


def allowed_file(filename):
    if '.' in filename:
        if filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS:
            return True
    return False


class Img2TxtWebApp(flask.Flask):
    def __init__(self):
        super().__init__('Img2Txt')
        self._img2txt_inference = Image2Text(
            vocabulary=ms_coco_vocabulary,
            save_path='img2txt.ckpt',
            inference_only=True,
        )
    def get_inference_result(self, image_id)
        # TODO: Return image-word coembedding.
        input_image_path = get_image_path(image_id, 'jpg')
        generated_sentence = self._img2txt_inference.generate_sentence(
            input_image,
        )
        return generated_sentence


def index():
    return flask.redirect(flask.url_for('config'))


def config():
    app = flask.current_app
    image_file = None
    if flask.request.method == 'POST':
        if 'image_file' not in flask.request.files:
            print('No image file in the html request form.')
            return redirect(flask.request.url)

        image_file = flask.request.files['image_file']
        if image_file.filename == '':
            print('No selected file')
            return redirect(flask.request.url)

        if image_file and allowed_file(image_file.filename):
            filename = secure_file(image_file.filename)
            image_id = str(uuid.uuid4())
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                '{}.jpg'.format(image_id)
            )
            image_file.save(file_path)
            return flask.render_template(
                'config.html',
                image_id=image_id,
            )
    else:
        # TODO: Load a default image file.
        return flask.render_template('config.html')


def show_results():
    app = flask.current_app
    image_id = flask.request.form['image_id']

    generated_sentence = app.get_inference_result(image_id)



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
        '/', 'index', index, methods=['GET'],
    )
    web_app.add_url_rule(
        '/config', 'config', config, methods=['GET', 'POST'],
    )
    web_app.add_url_rule(
        '/input_image/<image_id>', 'input_image', input_image, methods=['GET'],
    )
    web_app.add_url_rule(
        '/embedding/<embedding_id>', 'embedding', embedding, methods=['GET'],
    )
    web_app.add_url_rule(
        '/results', 'show_results', show_results, methods=['POST'],
    )

    return web_app


def get_image_path(upload_folder, image_id, ext='jpg'):
    return '{}/{}.{}'.format(upload_folder, image_id, ext)

