import os
import cv2
import pickle
import numpy as np
from facenet_pytorch import MTCNN
import torch

class ImageDataset:
    def __init__(self, project_directory, frames_resolution=(64, 64, 1), conf_threshold=0.8):
        self.project_directory = project_directory
        self.frames_resolution = frames_resolution
        self.conf_threshold = conf_threshold
        self.mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.emotions_count = 8
        self.emotions = np.zeros(self.emotions_count)

    def to_one_hot(self, number):
        arr = np.zeros((1, self.emotions_count), dtype=int)
        arr[0][number] = 1
        return arr[0]

    def convert_videos_to_frames(self, videos_folder, frames_folder):
        videos_fn = os.path.join(self.project_directory, videos_folder)
        frames_fn = os.path.join(self.project_directory, frames_folder)
        videos_list = os.listdir(videos_fn)

        for i, video_name in enumerate(videos_list):
            print(f"Progress: {i + 1}/{len(videos_list)}")
            video_path = os.path.join(videos_fn, video_name)
            emotion = int(video_name.split("-")[2]) - 1

            # Create directory for frames
            res_fn = os.path.join(frames_fn, str(i))
            os.makedirs(res_fn, exist_ok=True)

            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 0

            while success:
                frame_path = os.path.join(res_fn, f"frame_{count}.png")
                cv2.imwrite(frame_path, image)
                success, image = vidcap.read()
                count += 1

            annotations = {"emotion": emotion}
            with open(os.path.join(res_fn, "annotation.pickle"), 'wb') as f:
                pickle.dump(annotations, f, protocol=pickle.HIGHEST_PROTOCOL)

    def process_frames_to_pickle(self, frames_folder, save_folder, folder_start_index=0, skip=0, max_count=1000, frames_count=30, max_from_folder=1000, max_array_count=1000, save_index=500):
        data = []
        frames_folders = sorted(os.listdir(frames_folder), key=int)

        for i, folder_name in enumerate(frames_folders):
            if i < folder_start_index:
                continue

            if i % save_index == 0 and i != 0:
                with open(os.path.join(self.project_directory, save_folder), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            folder_path = os.path.join(frames_folder, folder_name)

            # Load emotion annotation
            with open(os.path.join(folder_path, "annotation.pickle"), 'rb') as handle:
                annotation = pickle.load(handle)

            emotion = annotation["emotion"]

            if os.path.exists(folder_path) and (len(os.listdir(folder_path)) - 1 >= frames_count):
                if self.emotions[emotion] < max_count:
                    length = len(os.listdir(folder_path)) - 1
                    array_count = 0
                    index = 0
                    images = []

                    while index < length and max_from_folder > array_count:
                        image_path = os.path.join(folder_path, f"frame_{index}.png")
                        image = cv2.imread(image_path, 1)

                        # Detect faces using MTCNN
                        boxes, _ = self.mtcnn.detect(image)

                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                detected_face = image[y1:y2, x1:x2]

                                if detected_face.size != 0:
                                    # Resize, normalize, and convert to grayscale if needed
                                    if self.frames_resolution[2] == 1:
                                        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                                    detected_face = cv2.resize(detected_face, (self.frames_resolution[0], self.frames_resolution[1]))
                                    detected_face = cv2.normalize(detected_face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                    images.append(detected_face)

                                    # Skip frames
                                    index += skip

                        index += 1

                        if len(images) == frames_count:
                            item = {"emotion": np.asarray(self.to_one_hot(emotion)), "images": np.asarray(images)}
                            images = []
                            array_count += 1
                            self.emotions[emotion] += 1
                            data.append(item)

                        if array_count >= max_array_count:
                            break

                    print(f"Processed: {i}/{len(frames_folders)} \tAdded: Emotion: {emotion} \t{array_count} arrays of {frames_count} images")
                else:
                    print(f"Processed: {i}/{len(frames_folders)} \tEmotion: {emotion} \tAlready enough data of this emotion")
            else:
                print("Directory doesn't exist or contains insufficient frames")

        return data

    def save_data(self, data, file_name):
        file_path = os.path.join(self.project_directory, "dataset", file_name)
        with open(file_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    project_dir = os.getcwd()
    processor = EmotionDataProcessor(project_dir)

    # Convert videos to frames
    processor.convert_videos_to_frames('dataset\\TrainingVideos', 'dataset\\TrainingFrames')
    processor.convert_videos_to_frames('dataset\\TestVideos', 'dataset\\TestFrames')

    # Process frames to pickle
    train_data = processor.process_frames_to_pickle('dataset\\TrainingFrames', 'dataset\\train_data.pickle')
    test_data = processor.process_frames_to_pickle('dataset\\TestFrames', 'dataset\\test_data.pickle')

    # Save processed data
    processor.save_data(train_data, "train_data.pickle")
    processor.save_data(test_data, "test_data.pickle")