import cv2
import mediapipe as mp
import fitz
import numpy as np
from math import hypot
import tkinter as tk
from PIL import Image, ImageTk
import os
import time
import tkinter.filedialog as filedialog


class ThreeFingerPDFController:
    def __init__(self):
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("All Files", "*.*")]
        )

        # Check if a file was selected
        if not pdf_path:
            print("No file selected. Exiting.")
            exit()

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize PDF
        self.pdf_document = fitz.open(pdf_path)
        self.current_page = 0
        self.total_pages = len(self.pdf_document)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Create GUI window
        self.root = tk.Tk()
        self.root.title("Gesture-Based PDF Controller with Drawing")

        # Drawing variables
        self.drawing_mode = "none"
        self.last_x = None
        self.last_y = None
        self.drawings = {}
        self.pen_color = "black"
        self.pen_width = 2
        self.eraser_width = 20

        self.setup_gui()

        # Gesture tracking variables
        self.gesture_start_time = None
        self.current_gesture = None
        self.gesture_cooldown = 0
        self.gesture_hold_time = 0.5

        # Status text
        self.status_text = ""
        self.status_cooldown = 0

        # Add pointer variables
        self.pointer = None
        self.pointer_size = 7
        self.pointer_color = "red"
        self.pointer_visible = True

    def update_pointer(self, x, y):
        x = x * self.canvas_width
        y = y * self.canvas_height

        # Delete existing pointer
        self.canvas.delete("pointer")

        # Create new pointer
        self.pointer = self.canvas.create_oval(
            x - self.pointer_size / 2,
            y - self.pointer_size / 2,
            x + self.pointer_size / 2,
            y + self.pointer_size / 2,
            fill=self.pointer_color,
            outline=self.pointer_color,
            tags="pointer"
        )

        # Create crosshair lines
        line_length = self.pointer_size * 1.5
        self.canvas.create_line(
            x - line_length, y,
            x + line_length, y,
            fill=self.pointer_color,
            tags="pointer"
        )
        self.canvas.create_line(
            x, y - line_length,
            x, y + line_length,
            fill=self.pointer_color,
            tags="pointer"
        )

    def setup_gui(self):
        # Main container
        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Video frame (left side)
        self.video_frame = tk.Frame(self.container)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Status label
        self.status_label = tk.Label(self.video_frame, text="", font=('Arial', 12))
        self.status_label.pack(pady=5)

        # PDF frame (right side)
        self.pdf_frame = tk.Frame(self.container)
        self.pdf_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Canvas for PDF and drawing
        self.canvas_width = 800
        self.canvas_height = 800
        self.canvas = tk.Canvas(self.pdf_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Page counter
        self.page_label = tk.Label(
            self.pdf_frame,
            text=f"Page: {self.current_page + 1}/{self.total_pages}",
            font=('Arial', 12)
        )
        self.page_label.pack(pady=5)

        # Mode label
        self.mode_label = tk.Label(
            self.pdf_frame,
            text="Mode: None",
            font=('Arial', 12)
        )
        self.mode_label.pack(pady=5)

    def detect_finger_gesture(self, hand_landmarks):

        # Finger tip landmarks
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Finger base landmarks
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]

        # Get thumb tip for additional control
        thumb_tip = hand_landmarks.landmark[4]

        # Check different finger combinations
        index_up = index_tip.y < index_base.y
        middle_up = middle_tip.y < middle_base.y
        ring_up = ring_tip.y < ring_base.y
        pinky_up = pinky_tip.y < pinky_base.y

        # Detect different gestures
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pen", True
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "eraser", True
        elif index_up and middle_up and ring_up and not pinky_up:
            return "next", True
        elif index_up and middle_up and ring_up and pinky_up:
            return "prev", True

        return None, False

    def get_finger_position(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        return index_tip.x, index_tip.y

    def handle_drawing(self, hand_landmarks):
        if self.drawing_mode in ["pen", "eraser"]:
            x, y = self.get_finger_position(hand_landmarks)

            self.update_pointer(x, y)

            canvas_x = x * self.canvas_width
            canvas_y = y * self.canvas_height

            if self.last_x is not None and self.last_y is not None:
                if self.drawing_mode == "pen":
                    line = self.canvas.create_line(
                        self.last_x, self.last_y, canvas_x, canvas_y,
                        fill=self.pen_color,
                        width=self.pen_width,
                        tags=f"page_{self.current_page}"
                    )
                    self.add_to_drawings(line)
                elif self.drawing_mode == "eraser":
                    # Erase items near the eraser position
                    items = self.canvas.find_overlapping(
                        canvas_x - self.eraser_width / 2,
                        canvas_y - self.eraser_width / 2,
                        canvas_x + self.eraser_width / 2,
                        canvas_y + self.eraser_width / 2
                    )
                    for item in items:
                        if self.canvas.gettags(item) and f"page_{self.current_page}" in self.canvas.gettags(item):
                            self.canvas.delete(item)
                            self.remove_from_drawings(item)

            self.last_x = canvas_x
            self.last_y = canvas_y
        else:
            self.last_x = None
            self.last_y = None

    def add_to_drawings(self, item):
        if self.current_page not in self.drawings:
            self.drawings[self.current_page] = set()
        self.drawings[self.current_page].add(item)

    def remove_from_drawings(self, item):
        if self.current_page in self.drawings:
            self.drawings[self.current_page].discard(item)

    def handle_gestures(self, hand_landmarks):
        gesture_type, gesture_detected = self.detect_finger_gesture(hand_landmarks)

        current_time = time.time()

        if gesture_detected:

            index_tip = hand_landmarks.landmark[8]
            index_base = hand_landmarks.landmark[5]
            if index_tip.y < index_base.y:
                x, y = self.get_finger_position(hand_landmarks)
                self.update_pointer(x, y)
                self.pointer_visible = True
            else:
                self.pointer_visible = False
                if self.pointer:
                    self.canvas.delete("pointer")

            # Handle drawing modes immediately
            if gesture_type in ["pen", "eraser"]:
                if self.drawing_mode != gesture_type:
                    self.drawing_mode = gesture_type
                    self.update_status(f"{gesture_type.capitalize()} mode activated")
                    self.mode_label.config(text=f"Mode: {gesture_type.capitalize()}")
                self.handle_drawing(hand_landmarks)
            else:
                # Handle page turning gestures with time delay
                if self.gesture_start_time is None:
                    self.gesture_start_time = current_time
                    self.current_gesture = gesture_type
                    self.update_status(f"Holding {'three' if gesture_type == 'next' else 'four'} fingers")
                elif self.current_gesture == gesture_type:
                    hold_duration = current_time - self.gesture_start_time

                    if hold_duration >= self.gesture_hold_time and self.gesture_cooldown <= 0:
                        if gesture_type == "next" and self.current_page < self.total_pages - 1:
                            self.current_page += 1
                            self.update_status("Next page")
                        elif gesture_type == "prev" and self.current_page > 0:
                            self.current_page -= 1
                            self.update_status("Previous page")

                        self.page_label.config(
                            text=f"Page: {self.current_page + 1}/{self.total_pages}"
                        )
                        self.gesture_start_time = None
                        self.gesture_cooldown = 10

                        self.refresh_canvas()
        else:
            self.gesture_start_time = None
            self.current_gesture = None

            self.pointer_visible = False
            self.canvas.delete("pointer")

    def refresh_canvas(self):
        self.canvas.delete("all")
        self.display_page()

        if self.current_page in self.drawings:
            pass

        if self.pointer_visible and self.last_x is not None and self.last_y is not None:
            self.update_pointer(self.last_x / self.canvas_width, self.last_y / self.canvas_height)

    def display_page(self):
        try:
            page = self.pdf_document[self.current_page]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Resize image to fit canvas
            img.thumbnail((self.canvas_width, self.canvas_height))

            photo = ImageTk.PhotoImage(image=img)

            # Store the image to prevent garbage collection
            self.current_image = photo

            # Create image on canvas
            self.canvas.create_image(
                self.canvas_width / 2,
                self.canvas_height / 2,
                image=photo,
                anchor=tk.CENTER
            )
        except Exception as e:
            print(f"Error displaying PDF page: {str(e)}")

    def update_status(self, text, duration=30):
        self.status_text = text
        self.status_cooldown = duration

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        if self.gesture_cooldown <= 0:
                            self.handle_gestures(hand_landmarks)

                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=photo)
                self.video_label.image = photo

                if self.status_cooldown > 0:
                    self.status_label.config(text=self.status_text)
                    self.status_cooldown -= 1
                else:
                    self.status_label.config(text="")

            self.root.after(10, self.update_frame)
        except Exception as e:
            print(f"Error updating frame: {str(e)}")

    def run(self):
        self.update_frame()
        self.root.mainloop()

    def cleanup(self):
        self.cap.release()
        self.pdf_document.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        controller = ThreeFingerPDFController()

        controller.run()

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        if 'controller' in locals():
            controller.cleanup()
