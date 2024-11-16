import streamlit as st
import cv2
import mediapipe as mp
import fitz
import numpy as np
from PIL import Image
import io
import time
import tempfile
from pathlib import Path


class GesturePDFController:
    def __init__(self):
        # Initialize MediaPipe

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        if 'drawings' not in st.session_state:
            st.session_state.drawings = {}
        if 'drawing_mode' not in st.session_state:
            st.session_state.drawing_mode = "none"
        if 'pdf_document' not in st.session_state:
            st.session_state.pdf_document = None
        if 'last_gesture' not in st.session_state:
            st.session_state.last_gesture = None
        if 'gesture_time' not in st.session_state:
            st.session_state.gesture_time = time.time()
        if 'drawing_points' not in st.session_state:
            st.session_state.drawing_points = {}
        if 'current_path' not in st.session_state:
            st.session_state.current_path = []
        if 'cursor_position' not in st.session_state:
            st.session_state.cursor_position = (0, 0)
        if 'pdf_dimensions' not in st.session_state:
            st.session_state.pdf_dimensions = (0, 0)

        # Drawing settings
        self.pen_color = (255, 0, 0)  # RGB for red
        self.pen_width = 2
        self.eraser_width = 20
        self.gesture_hold_time = 0.5
        self.gesture_cooldown = 0.5
        self.crosshair_size = 20
        self.crosshair_color = (0, 255, 0)  # RGB for green

    def draw_crosshair(self, img, position):
        x, y = position
        # Draw horizontal line
        cv2.line(img,
                 (x - self.crosshair_size, y),
                 (x + self.crosshair_size, y),
                 self.crosshair_color,
                 2)
        # Draw vertical line
        cv2.line(img,
                 (x, y - self.crosshair_size),
                 (x, y + self.crosshair_size),
                 self.crosshair_color,
                 2)
        return img

    def get_scaled_cursor_position(self, hand_landmarks, frame_shape):
        index_tip = hand_landmarks.landmark[8]
        camera_width, camera_height = frame_shape[1], frame_shape[0]
        pdf_width, pdf_height = st.session_state.pdf_dimensions

        # Get position in camera coordinates (0-1)
        x_percent = index_tip.x
        y_percent = index_tip.y

        # Scale to PDF dimensions
        pdf_x = int(x_percent * pdf_width)
        pdf_y = int(y_percent * pdf_height)

        return (pdf_x, pdf_y)

    def update_drawing(self, page_num, position):
        if page_num not in st.session_state.drawing_points:
            st.session_state.drawing_points[page_num] = []

        if st.session_state.drawing_mode == "pen":
            st.session_state.current_path.append(position)
        elif st.session_state.drawing_mode == "eraser":
            if st.session_state.drawing_points[page_num]:
                for path in st.session_state.drawing_points[page_num][:]:  # Create a copy of the list to modify
                    for point in path:
                        if np.linalg.norm(np.array(position) - np.array(point)) < self.eraser_width:
                            st.session_state.drawing_points[page_num].remove(path)
                            break

    def finalize_path(self, page_num):
        if st.session_state.current_path and page_num is not None:
            if page_num not in st.session_state.drawing_points:
                st.session_state.drawing_points[page_num] = []
            if len(st.session_state.current_path) > 1:
                st.session_state.drawing_points[page_num].append(st.session_state.current_path.copy())
            st.session_state.current_path = []

    def draw_on_image(self, img_array, cursor_pos=None):
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        img = img_array.copy()

        page_num = st.session_state.current_page
        if page_num in st.session_state.drawing_points:
            for path in st.session_state.drawing_points[page_num]:
                if len(path) > 1:
                    path_array = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [path_array], False, self.pen_color, self.pen_width)

        if st.session_state.current_path and len(st.session_state.current_path) > 1:
            path_array = np.array(st.session_state.current_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [path_array], False, self.pen_color, self.pen_width)

        # Draw crosshair if cursor position is provided
        if cursor_pos is not None and st.session_state.drawing_mode in ["pen", "eraser"]:
            img = self.draw_crosshair(img, cursor_pos)

        return img

    def display_page(self, pdf_placeholder):
        if st.session_state.pdf_document is not None:
            page = st.session_state.pdf_document[st.session_state.current_page]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            # Store PDF dimensions
            st.session_state.pdf_dimensions = (pix.width, pix.height)

            # Convert to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            # Draw on the image, including the cursor position
            img_with_drawings = self.draw_on_image(img, st.session_state.cursor_position)

            # Convert to PIL Image
            img_pil = Image.fromarray(img_with_drawings)

            # Convert to bytes for display
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Display image
            pdf_placeholder.image(img_byte_arr, use_column_width=True)

    def setup_interface(self):
        st.title("Gesture-Controlled PDF Viewer with Drawing")

        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if uploaded_file is not None:
            if st.session_state.pdf_document is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.pdf_document = fitz.open(tmp_file.name)
                    st.session_state.total_pages = len(st.session_state.pdf_document)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Camera Feed")
                camera_placeholder = st.empty()
                status_placeholder = st.empty()

                st.markdown("""
                ### Gestures Guide:
                - ☝️ One finger up: Pen mode
                - ✌️ Two fingers up: Eraser mode
                - 🖐️ Three fingers up: Next page
                - 👋 Four fingers up: Previous page
                """)

            with col2:
                st.markdown("### PDF View")
                pdf_placeholder = st.empty()
                st.markdown(f"Page: {st.session_state.current_page + 1}/{st.session_state.total_pages}")
                st.markdown(f"Mode: {st.session_state.drawing_mode.capitalize()}")

            return camera_placeholder, status_placeholder, pdf_placeholder

        return None, None, None

    def detect_finger_gesture(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]

        index_up = index_tip.y < index_base.y
        middle_up = middle_tip.y < middle_base.y
        ring_up = ring_tip.y < ring_base.y
        pinky_up = pinky_tip.y < pinky_base.y

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pen"
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "eraser"
        elif index_up and middle_up and ring_up and not pinky_up:
            return "next"
        elif index_up and middle_up and ring_up and pinky_up:
            return "prev"

        return None

    def process_gesture(self, gesture, status_placeholder):
        current_time = time.time()

        if gesture != st.session_state.last_gesture:
            self.finalize_path(st.session_state.current_page)
            st.session_state.last_gesture = gesture
            st.session_state.gesture_time = current_time

        if gesture in ["pen", "eraser"]:
            st.session_state.drawing_mode = gesture
            status_placeholder.markdown(f"Mode: {gesture.capitalize()}")

        elif current_time - st.session_state.gesture_time >= self.gesture_hold_time:
            if gesture == "next" and st.session_state.current_page < st.session_state.total_pages - 1:
                self.finalize_path(st.session_state.current_page)
                st.session_state.current_page += 1
                status_placeholder.markdown("Next page")

            elif gesture == "prev" and st.session_state.current_page > 0:
                self.finalize_path(st.session_state.current_page)
                st.session_state.current_page -= 1
                status_placeholder.markdown("Previous page")

            st.session_state.gesture_time = current_time

    def run(self):
        camera_placeholder, status_placeholder, pdf_placeholder = self.setup_interface()

        if all([camera_placeholder, status_placeholder, pdf_placeholder]):
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        # Get index finger position scaled to PDF dimensions
                        cursor_pos = self.get_scaled_cursor_position(hand_landmarks, frame.shape)
                        st.session_state.cursor_position = cursor_pos

                        gesture = self.detect_finger_gesture(hand_landmarks)
                        if gesture:
                            self.process_gesture(gesture, status_placeholder)

                        # Update drawing if in drawing mode
                        if st.session_state.drawing_mode in ["pen", "eraser"]:
                            self.update_drawing(st.session_state.current_page, cursor_pos)
                else:
                    self.finalize_path(st.session_state.current_page)

                # Display camera feed
                camera_placeholder.image(rgb_frame, channels="RGB", width=400)

                # Display PDF page with drawings and crosshair
                self.display_page(pdf_placeholder)

                time.sleep(0.1)


def main():
    st.set_page_config(layout="wide")
    controller = GesturePDFController()
    controller.run()


if __name__ == "__main__":
    main()