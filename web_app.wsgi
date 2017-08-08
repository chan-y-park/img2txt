#!/usr/bin/env python

import flask

def get_test_web_app():
    web_app = flask.Flask()
    web_app.config.update(DEBUG=True)
    web_app.add_url_rule(
        '/', 'index', index, methods=['GET'],
    )

def index():
    return flask.render_template('index.html')
