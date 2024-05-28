import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import base64

from src.utils.streamlitUtils import *

image = set_streamlit_header()
openai_client, detector, brush_thick, eraser_thick, rectKernel, options, counter_map, blkboard = set_basic_config()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


prompt = generate_user_prompt()


def callback(frame, xp=0, yp=0):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:

        _, x1, y1 = landmark_list[8]  # Tip of Index Finger
        _, x2, y2 = landmark_list[12]  # Tip of Middle Finger
        fingers = detector.fingers()

        if len(fingers) == 5:
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                cv2.rectangle(img, (x1 - 25, y1 - 25), (x2 + 25, y2 + 25), (0, 0, 255), cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(blkboard, (xp, yp), (x1, y1), (0, 0, 0), eraser_thick)
                counter_map["go"] = 0
                xp, yp = x1, y1

            elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[
                4] == 0:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                print("write")
                counter_map["go"] = 0
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(blkboard, (xp, yp), (x1, y1), (0, 255, 0), brush_thick)
                xp, yp = x1, y1

            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[
                4] == 0:
                xp, yp = 0, 0
                print("Go")

                blackboard_gray = cv2.cvtColor(blkboard, cv2.COLOR_RGB2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
                if len(blackboard_cnts) >= 1:
                    bounding_boxes = []
                    for cnt in sorted(blackboard_cnts, key=cv2.contourArea, reverse=True):
                        if cv2.contourArea(cnt) > 800:
                            x, y, w, h = cv2.boundingRect(cnt)
                            bounding_boxes.append((x, y))
                            bounding_boxes.append((x + w, y + h))
                    box = cv2.minAreaRect(np.asarray(bounding_boxes))
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(box)
                    a1 = min(x1, x2, x3, x4)
                    a2 = max(x1, x2, x3, x4)
                    b1 = min(y1, y2, y3, y4)
                    b2 = max(y1, y2, y3, y4)
                    cv2.rectangle(img, (int(a1), int(b1)), (int(a2), int(b2)), (0, 255, 0), 2)
                    digit = blackboard_gray
                    print(digit)
                    counter_map["go"] += 1
                    if counter_map["go"] > 20:
                        result_queue.put(True)
                        counter_map["go"] = 0
                        cv2.imwrite("math.png", digit)
            else:
                xp, yp = 0, 0
                counter_map["go"] = 0

    gray = cv2.cvtColor(blkboard, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    if img.shape[0] == 720 and img.shape[1] == 1280:
        print("Resolution Achieved")
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, blkboard)

    return av.VideoFrame.from_ndarray(img)


result_queue = (
    queue.Queue()
)
col1, col2 = st.columns([6, 6])

with col1:
    with st.container(height=570):
        ctx = webrtc_streamer(key="socratic",
                              mode=WebRtcMode.SENDRECV,
                              async_processing=True,
                              rtc_configuration={
                                  "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                              },
                              video_frame_callback=callback,
                              media_stream_constraints={"video":
                                                            {"width": {"ideal": 1280, "min": 1280},
                                                             "height": {"ideal": 720, "min": 720}},
                                                        "audio": False,
                                                        }
                              )
with col2:
    if ctx is not None:
        with st.container(height=570):
            col1, col2, col3 = st.columns([4, 2, 4])
            with col1:
                st.write(' ')
            with col2:
                st.markdown("""## Socratic""")
            with col3:
                st.write(' ')
            while ctx.state.playing:
                try:
                    result = result_queue.get()
                    if "messages" not in st.session_state and result:
                        st.session_state.messages = []

                    with st.chat_message("user", avatar="👩‍🚀"):
                        st.write_stream(response_generator(prompt, 0.01))
                        col1, col2, col3 = st.columns([2, 3, 3])
                        with col1:
                            st.write(' ')
                        with col2:
                            st.image('math.png', width=300, caption='Image submitted by user')
                        with col3:
                            st.write(' ')

                    if result:
                        with st.chat_message("assistant", avatar="🦉"):
                            base64_image = encode_image("math.png")
                            stream = openai_client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": "system",
                                     "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Solve this mathematical image in markdown for me?"},
                                        {"type": "image_url", "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"}
                                         }
                                    ]}
                                ],
                                stream=True,
                            )
                            response = st.write_stream(stream)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        result_queue.queue.clear()
                except queue.Empty:
                    result = None

set_streamlit_footer()
