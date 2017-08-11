#!/usr/bin/env python

import sys
import os
import getopt

import flask
from flask import Flask

from web_ui import get_web_app 

def index():
    return flask.render_template('index.html')

def test():
    return flask.render_template('test.html')

def get_test_web_app():
    web_app = Flask(
        'test',
        template_folder=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'templates',
        ),
    )
    web_app.config.update(DEBUG=True)
    web_app.add_url_rule(
        '/', 'index', index, methods=['GET'],
    )
    web_app.add_url_rule(
        '/test', 'test', test, methods=['GET'],
    )
    return web_app

application = get_web_app()

if __name__ == '__main__':
    host = '0.0.0.0'
    port = '9999'

    opts, args = getopt.getopt(sys.argv[1:], 'p:')
    for opt, arg in opts:
        if opt == '-p':
            port = int(arg)

    application.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False,
        threaded=True,
    )
