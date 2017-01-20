import websocket
from threading import Thread
import base64
import os


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        dir = 'IMAGES_FOR_TEST'
        fileSet = set()

        for dir_, _, files in os.walk(dir):
            for fileName in files:
                relDir = os.path.relpath(dir_, dir)
                relFile = os.path.join(relDir, fileName)
                fileSet.add(relFile)
        for test_image in fileSet:
            with open(dir+'/'+test_image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                ws.send(encoded_string)

    Thread(target=run).start()


if __name__ == "__main__":

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://127.0.0.1:8888",
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
