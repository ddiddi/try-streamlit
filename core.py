# -*- coding: utf-8 -*-
# Neo Combat Labs - Jiu-Jitsu Match Analysis

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

def main():
    # Render instructions (Jiu-Jitsu/Neo Combat Labs themed).
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies (model files).
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Sidebar mode selection.
    st.sidebar.title("Neo Combat Labs: Jiu-Jitsu Analysis")
    app_mode = st.sidebar.selectbox(
        "Select an Operation Mode",
        ["Show Instructions", "Analyze Jiu-Jitsu Match", "Run ID Matching"]
    )
    if app_mode == "Show Instructions":
        st.sidebar.success('To begin, select "Analyze Jiu-Jitsu Match" or "Run ID Matching".')
    elif app_mode == "Analyze Jiu-Jitsu Match":
        readme_text.empty()
        run_the_app()
    elif app_mode == "Run ID Matching":
        readme_text.empty()
        run_id_matching_app()

def download_file(file_path):
    # Avoid re-downloading files if they already exist.
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning(f"Downloading {file_path} from training HQ...")
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    weights_warning.warning(
                        f"Downloading {file_path}... ({counter / MEGABYTES:6.2f}/{length / MEGABYTES:6.2f} MB)"
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def run_the_app():
    # Optional user-uploaded footage of a Jiu-Jitsu match.
    video_file = st.sidebar.file_uploader("Upload Jiu-Jitsu Match Footage", type=["mp4","avi","mov","mkv"])
    
    # Load metadata and create summary.
    def load_metadata(url):
        return pd.read_csv(url)

    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_pedestrian": "grappler",
            "label_biker": "coach",
            "label_car": "competing_fighter",
            "label_trafficLight": "referee",
            "label_truck": "staff_member"
        })
        return summary

    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    # UI to select frame.
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index is None:
        st.error("No frames match your criteria. Adjust your selection.")
        return

    # YOLO model parameters.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load frame: either from user-uploaded match footage or the demo dataset.
    if video_file is not None:
        video_bytes = video_file.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture("temp_video.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if selected_frame_index < 0 or selected_frame_index >= total_frames:
            st.error("Selected frame index is out of range for the uploaded footage.")
            cap.release()
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Could not read the selected frame from the footage.")
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Use the dataset image if no footage is uploaded.
        image_url = os.path.join(DATA_URL_ROOT, selected_frame)
        image = load_image(image_url)

    # Ground truth boxes from the metadata.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])

    # Perform YOLO detection.
    yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)

    # Two tabs: one for scanning the environment, one for identifying a grappler.
    tab1, tab2 = st.tabs(["Match Overview", "Identify Grappler"])

    with tab1:
        draw_image_with_boxes(
            image,
            boxes,
            "Neo Combat Labs Ground Truth",
            f"**Verified Match Data** (frame `{selected_frame_index}`)"
        )
        draw_image_with_boxes(
            image,
            yolo_boxes,
            "Neo Combat Labs Real-Time Detection",
            f"**Jiu-Jitsu Model** (overlap `{overlap_threshold:.1f}`) (confidence `{confidence_threshold:.1f}`)"
        )

    with tab2:
        st.write("Select a grappler to run identification:")
        grapplers = yolo_boxes[yolo_boxes['labels'] == 'grappler']

        if len(grapplers) == 0:
            st.write("No grapplers detected in this frame.")
        else:
            grappler_index = st.selectbox("Select grappler index", range(len(grapplers)))
            selected_grappler = grapplers.iloc[grappler_index]

            st.write("Selected grappler details:", selected_grappler)
            if st.button("Run ID on selected grappler"):
                st.write("Initiating grappler identification protocol...")
                # Placeholder for ID logic.
                st.write("Grappler ID established. (Placeholder)")

def run_id_matching_app():
    st.header("Run ID Matching")
    st.write("Upload an image or video of a Jiu-Jitsu scenario and run ID matching on detected grapplers.")

    file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi","mov","mkv"])
    if file is not None:
        file_type = file.name.split('.')[-1].lower()
        is_video = file_type in ["mp4","avi","mov","mkv"]

        if is_video:
            # Handle video
            with open("temp_id_video.mp4", "wb") as f:
                f.write(file.read())
            cap = cv2.VideoCapture("temp_id_video.mp4")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_to_analyze = st.slider("Select frame to analyze", 0, max(total_frames-1, 0), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_analyze)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Could not read the selected frame from the uploaded video.")
                return

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Handle image
            file_bytes = np.asarray(bytearray(file.read()), dtype="uint8")
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = img[:, :, [2, 1, 0]]  # BGR to RGB

        # After loading the image, run YOLO detection
        confidence_threshold, overlap_threshold = object_detector_ui_id_mode()

        yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)

        st.subheader("Detections")
        draw_image_with_boxes(
            image,
            yolo_boxes,
            "Detected Entities",
            f"Confidence: {confidence_threshold:.2f}, Overlap: {overlap_threshold:.2f}"
        )

        # Allow user to select a grappler and run ID on them
        grapplers = yolo_boxes[yolo_boxes['labels'] == 'grappler']
        if len(grapplers) > 0:
            st.write("Select a grappler to match ID:")
            grappler_index = st.selectbox("Select grappler index", range(len(grapplers)))
            selected_grappler = grapplers.iloc[grappler_index]
            st.write("Selected grappler:", selected_grappler)
            if st.button("Run ID Matching"):
                st.write("Performing ID matching on the selected grappler...")
                # Insert ID matching logic here
                st.write("ID Match complete! (Placeholder)")
        else:
            st.write("No grapplers detected to run ID matching on.")

def frame_selector_ui(summary):
    st.sidebar.markdown("## Frame Selection")

    object_type = st.sidebar.selectbox("Select Role", summary.columns, 2)
    min_elts, max_elts = st.sidebar.slider(f"Number of {object_type}s (range)?", 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None

    selected_frame_index = st.sidebar.slider("Select Frame Index", 0, len(selected_frames) - 1, 0)
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y(f"{object_type}:Q")
    )
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x="selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

def object_detector_ui():
    st.sidebar.markdown("## Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap Threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def object_detector_ui_id_mode():
    st.markdown("### ID Matching Detection Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.slider("Overlap Threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def draw_image_with_boxes(image, boxes, header, description):
    # Updated colors and classes for Jiu-Jitsu context:
    LABEL_COLORS = {
        "competing_fighter": [200, 20, 20],
        "grappler": [20, 200, 20],
        "staff_member": [20, 20, 200],
        "referee": [200, 200, 20],
        "coach": [200, 20, 200],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        color = LABEL_COLORS.get(label, [255,255,255])
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += color
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

def yolo_v3(image, confidence_threshold, overlap_threshold):
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names

    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box_vals = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box_vals.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map YOLO IDs to Jiu-Jitsu context:
    NEO_BJJ_LABELS = {
        0: 'grappler',          # pedestrian
        1: 'coach',             # biker
        2: 'competing_fighter', # car
        3: 'coach',             # biker (again)
        5: 'staff_member',      # truck
        7: 'staff_member',      # truck (again)
        9: 'referee'            # traffic light
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            label = NEO_BJJ_LABELS.get(class_IDs[i], None)
            if label is None:
                continue
            x, y, w, h = boxes[i]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x + w)
            ymax.append(y + h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

if __name__ == "__main__":
    main()
