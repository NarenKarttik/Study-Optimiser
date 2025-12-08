import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import timm 

class FERModel(nn.Module):
    def __init__(self, model_arch: str, pretrained: bool, num_classes: int = 7):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

class FaceMeshAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        return results

    def draw_landmarks(self, frame, pose_results):
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

class FerAnalyzer:
    FER_CATEGORIES = [
        'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear'
    ]

    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading FER model on device: {self.device}")
        self.model = self._load_model(model_path)
        if not self.model:
            raise RuntimeError("Failed to load FER model.")
        else:
            print("FER model (ResNet34) loaded successfully.")

    def _load_model(self, model_path):
        try:
            model = FERModel(model_arch='resnet34', pretrained=False, num_classes=7)
            cp = torch.load(model_path, map_location=self.device)
            state_dict = cp["state_dict"]
            state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()  
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def predict(self, preprocessed_face_tensor):
        if not self.model:
            print("Model not loaded, cannot predict.")
            return None
            
        tensor = preprocessed_face_tensor.to(self.device)
        
        with torch.no_grad():
            output_logits = self.model(tensor)
        
        probabilities = torch.softmax(output_logits, dim=1).squeeze().cpu().numpy()
        
        return dict(zip(self.FER_CATEGORIES, probabilities))