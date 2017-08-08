#!/usr/bin/env python

import sys
import getopt
import flask

def get_test_web_app():
    web_app = flask.Flask(
        'test',
        template_folder='templates'    
    )
    web_app.config.update(DEBUG=True)
    web_app.add_url_rule(
        '/', 'index', index, methods=['GET'],
    )
    return web_app

def index():
    return flask.render_template('index.html')

application = get_test_web_app()

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
