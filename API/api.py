from flask import Flask, request, jsonify
import numpy as np
import cv2
import io
from PIL import Image
import json
import base64
import logging

from .enums import HttpStatusCodes
from sudoku_solver.solver import solve
from utils.helpers import *

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route("/get", methods=["GET"])
def get_hello_world():
    return jsonify({"result": "Hello World"})

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'}), HttpStatusCodes.BAD_REQUEST

    file = request.files['file']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'}), HttpStatusCodes.BAD_REQUEST

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = np.array(image)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), HttpStatusCodes.BAD_REQUEST

    height_img = 450
    width_img = 450
    model = initialize_prediction_model()

    img = cv2.resize(image, (width_img, height_img))
    img_blank = np.zeros((height_img, width_img, 3), np.uint8)
    img_threshold = preprocess(img)

    img_contours = img.copy()
    img_big_contour = img.copy()
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    biggest, _ = biggest_contour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(img_big_contour, biggest, -1, (0, 0, 255), 25)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp_colored = cv2.warpPerspective(img, matrix, (width_img, height_img))
        img_warp_colored = cv2.bitwise_not(img_warp_colored)
        img_warp_colored = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)

        img_detected_digits = img_blank.copy()
        img_solved_digits = img_blank.copy()
        boxes = split_boxes(img_warp_colored)
        numbers = get_prediction(boxes, model)

        img_detected_digits = display_numbers(img_detected_digits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        pos_array = np.where(numbers > 0, 0, 1)

        board = np.array_split(numbers, 9)
        try:
            solve(board)
        except Exception as e:
            logging.error(f"Error solving sudoku: {e}")
            return jsonify({'error': 'Error solving sudoku'}), HttpStatusCodes.INT_SERVER_ERROR

        flat_list = [item for sublist in board for item in sublist]
        solved_numbers = flat_list * pos_array
        img_solved_digits = display_numbers(img_solved_digits, solved_numbers)

        pts2 = np.float32(biggest)
        pts1 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_in_warp_colored = cv2.warpPerspective(img_solved_digits, matrix, (width_img, height_img))
        inv_perspective = cv2.addWeighted(img_in_warp_colored, 1, img, 0.5, 1)
        img_detected_digits = draw_grid(img_detected_digits)
        img_solved_digits = draw_grid(img_solved_digits)

        _, img_encoded = cv2.imencode('.png', inv_perspective)
        img_bytes = io.BytesIO(img_encoded)

        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        numbers_json = json.dumps(numbers.tolist())
        solved_numbers_json = json.dumps(solved_numbers.tolist())

        return jsonify({'solution': numbers_json, 'image': img_base64, 'solved_numbers': solved_numbers_json})

    else:
        return jsonify({'error': 'No Sudoku Found'}), HttpStatusCodes.NOT_FOUND

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
