# magic-box server file
import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import json
from tornado.options import define, options
import bl
from send_results import send_results_to_football_fork


define("port", default=80, help="run on the given port", type=int)
define("csvfile", default='test/csv/events.csv', help="csv file", type=str)
define("csvfileloto", default='test/csv/loto_events.csv', help="csv file test", type=str)
define("save_result", default=False, help="Save results to json files", type=bool)
define("check_date", default=False, help="Check if magic-box support matches date", type=bool)


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/send_results", SendResultsHandler),
            (r"/", ChatSocketHandler),
        ]
        settings = dict(
            cookie_secret="NkNncTi5R6KQFP0OT4av9YW2gZtDDUxxifxKxFhhONw",
            xsrf_cookies=True,
        )
        super(Application, self).__init__(handlers, **settings)


class SendResultsHandler(tornado.web.RequestHandler):
    def get(self):
        send_results_to_football_fork()


class ChatSocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()  # server active client list

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):  # when client open connection with server this method executes
        ChatSocketHandler.waiters.add(self)
        self.write_message(json.dumps({
            'status': 'Connection open'
        }))

    def on_close(self):  # when client close
        ChatSocketHandler.waiters.remove(self)

    def on_message(self, message):  # when client send message(image) to server
        logging.info("got message")
        image_path, image_id = bl.save_image(message)
        self.write_message(json.dumps({
            'status': 'Image processing',
            'id': image_id
        }))
        result = None
        try:
            result = bl.parse_image(
                image_path,
                options.csvfile,
                options.csvfileloto,
                options.check_date,
            )
        except Exception as e:
            result = [{
                'status': 'Error',
                'message': str(e)
            }]
        if len(result) > 1:
            for message in result:
                self.write_message(json.dumps(message))
        else:
            self.write_message(json.dumps(result[0]))
            result = result[0]
        if options.save_result:
            logging.info("Saving result")
            try:
                bl.save_image_result(image_path, result)
            except Exception as e:
                logging.error("Error during saving: %s" % e)
        logging.info("Finished")


def main():
    tornado.options.parse_command_line()
    app = Application()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
