import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import mediapipe as mp
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


 


def process_video_with_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {video_path}")
        return None

    mp_pose = mp.solutions.pose # type: ignore
    pose = mp_pose.Pose(static_image_mode=False)
    frame_count = 0
    landmarks_all_frames = []

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks_all_frames.append(frame_landmarks)

        frame_count += 1
        progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

    cap.release()
    pose.close()

    st.success(f"Pose extraction complete. Total frames processed: {frame_count}")
    return np.array(landmarks_all_frames)  # shape: (num_frames, 33, 3)

def main():
    your_videos= [];
    st.set_page_config(page_title="Golfer's Pro", page_icon="⛳️", layout="centered")
    st.title("Golf Swing Video Analyzer")
    st.write("Upload a video of your golf swing for analysis.")
    your_swing = st.file_uploader("Upload a video of your swing" , type=["mp4","mov","avi"])
    pro_swing = st.file_uploader("Upload a video of a swing you want to emulate", type=["mp4", "mov", "avi"])

    if your_swing is not None:
        # Create a temporary file to save the uploaded video
        # This is necessary because cv2.VideoCapture often needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(your_swing.read())
            your_video_path = tmp_file.name
    if pro_swing is not None:
        # Create a temporary file to save the uploaded video
        # This is necessary because cv2.VideoCapture often needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(pro_swing.read())
            pro_video_path = tmp_file.name

        st.video(your_video_path) # Display the uploaded video for verification
        st.video(pro_video_path)

        st.write("Video uploaded successfully! Click 'Analyze Swing' to start processing.")

        if st.button("Analyze Swing") and your_video_path is not None and pro_video_path is not None:
            # Call your video processing function
            # In a real application, this would involve your MediaPipe pose estimation
            # and subsequent AI model analysis as described in the 'golf-ai-model' doc.
            you_processed_data = process_video_with_opencv(your_video_path)
            pro_processed_data = process_video_with_opencv(pro_video_path)

            # Example of how you might use the data from the 'golf-ai-model' doc:
            # from your_golf_ai_module import process_video_for_features, train_swing_model, analyze_new_swing
            # features, _ = process_video_for_features(video_path)
            # if features.size > 0:
            #     st.write("Extracted features from video. Now you can feed them to your AI model.")
            #     # Assuming you have a trained model
            #     # feedback = analyze_new_swing(your_trained_model, features[0]) # Analyze first frame for example
            #     # st.write(feedback)
            # else:
            #     st.warning("No pose landmarks detected in the video.")
            try:
                if you_processed_data is None or pro_processed_data is None:
                    st.warning("No pose landmarks detected in one or both videos.")
                    return

                # Convert pose data to summaries or downsampled format
                you_summary = you_processed_data[::10].tolist()  # take every 10th frame
                pro_summary = pro_processed_data[::10].tolist()

                prompt = f"""
                I have two golf swing pose data arrays:
                - My swing: {you_summary}
                - Pro swing: {pro_summary}

                Each array contains pose landmark coordinates in the format [x, y, z] for each of 33 body parts over time.

                Please analyze the differences and give me specific coaching advice on how to make my swing more like the pro's.
                Provide individualized, concise, and constructive feedback.
                """

                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a world renowned golf coach and biomechanical expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1000
                )

                st.markdown("### Individual Advice Here:")
                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"An error has occurred: {e}")

                # Clean up the temporary file after processing
            os.unlink(your_video_path)
            os.unlink(pro_video_path)
            # st.write("Temporary video file deleted.")

if __name__ == "__main__":
    main()
