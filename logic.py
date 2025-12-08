import numpy as np
import cv2

LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 145]
HPE_IDXS = [33, 263, 1, 61, 291, 152]
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],      # Nose tip
    [0.0, -330.0, -65.0],  # Chin
    [-225.0, 170.0, -135.0], # Left eye left corner
    [225.0, 170.0, -135.0],  # Right eye right corner
    [-150.0, -150.0, -125.0],# Left mouth corner
    [150.0, -150.0, -125.0]  # Right mouth corner
], dtype=np.float64)


class StudyOptimizerLogic:
    def __init__(self):
        self.EAR_THRESHOLD = 0.21
        self.EAR_CONSEC_FRAMES = 45 # ~3s at 15fps
        self.HPE_YAW_THRESHOLD = 25
        self.HPE_PITCH_THRESHOLD = 20
        self.SLOUCH_THRESHOLD = 0.025
        self.LEAN_THRESHOLD = 0.1
        self.POSTURE_CONSEC_FRAMES = 450 # ~30s
        self.DISTRACTION_CONSEC_FRAMES = 450 # ~30s
        self.HPE_SMOOTHING = 0.6
        
        self.TAKE_A_BREAK_THRESHOLD = 3 
        
        self.eye_closed_counter = 0
        self.distracted_counter = 0
        self.bad_posture_counter = 0
        self.alert_repeat_counter = 0
        
        self.alert_status = "None"
        self.current_status = "Initializing..."
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
    
    def _calculate_ear(self, eye_landmarks, frame_shape):
        h, w = frame_shape
        coords = np.array([(int(eye_landmarks[i].x * w), int(eye_landmarks[i].y * h)) for i in range(6)])
        p2_p6 = np.linalg.norm(coords[1] - coords[5])
        p3_p5 = np.linalg.norm(coords[2] - coords[4])
        p1_p4 = np.linalg.norm(coords[0] - coords[3])
        if p1_p4 == 0:
            return 0.0
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear

    def _get_head_pose(self, face_landmarks, frame_shape):
        h, w = frame_shape
        image_points_2d = np.array([
            (int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)) for idx in HPE_IDXS
        ], dtype=np.float64)
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            MODEL_POINTS_3D, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rotation_matrix[2,1] , rotation_matrix[2,2])
            yaw = np.arctan2(-rotation_matrix[2,0], sy)
            roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            pitch = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            yaw = np.arctan2(-rotation_matrix[2,0], sy)
            roll = 0
        
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def _calculate_posture(self, pose_landmarks, frame_shape):
        h, w = frame_shape
        
        if not pose_landmarks:
            return 0, "No Pose"

        try:
            if (pose_landmarks[LEFT_SHOULDER].visibility < 0.5 or 
                pose_landmarks[RIGHT_SHOULDER].visibility < 0.5):
                return 50, "Partial Pose"

            nose_y = pose_landmarks[NOSE].y
            ls_y = pose_landmarks[LEFT_SHOULDER].y
            rs_y = pose_landmarks[RIGHT_SHOULDER].y
            shoulder_avg_y = (ls_y + rs_y) / 2
            slouch_diff = nose_y - shoulder_avg_y
            
            nose_x = pose_landmarks[NOSE].x
            ls_x = pose_landmarks[LEFT_SHOULDER].x
            rs_x = pose_landmarks[RIGHT_SHOULDER].x
            shoulder_avg_x = (ls_x + rs_x) / 2
            lean_diff = abs(nose_x - shoulder_avg_x)
            
            # print(f"\r[DEBUG] Slouch Diff: {slouch_diff:+.4f} (Threshold: {self.SLOUCH_THRESHOLD}) | Lean Diff: {lean_diff:.4f} (Threshold: {self.LEAN_THRESHOLD})", end="")

            if slouch_diff > self.SLOUCH_THRESHOLD:
                return 0, "Slouching"
            elif lean_diff > self.LEAN_THRESHOLD:
                return 0, "Leaning"
            else:
                return 100, "Good"
                
        except Exception as e:
            return 0, "Error"

    def update_metrics(self, frame, face_landmarks, pose_landmarks, emotion_scores):
        frame_shape = frame.shape[:2]
        
        if face_landmarks:
            left_eye_lms = [face_landmarks[i] for i in LEFT_EYE_IDXS]
            right_eye_lms = [face_landmarks[i] for i in RIGHT_EYE_IDXS]
            ear = (self._calculate_ear(left_eye_lms, frame_shape) + self._calculate_ear(right_eye_lms, frame_shape)) / 2.0
            
            if ear < self.EAR_THRESHOLD:
                self.eye_closed_counter += 1
            else:
                self.eye_closed_counter = 0
            
            raw_yaw, raw_pitch, roll = self._get_head_pose(face_landmarks, frame_shape)
            self.smooth_yaw = (raw_yaw * (1.0 - self.HPE_SMOOTHING)) + (self.smooth_yaw * self.HPE_SMOOTHING)
            self.smooth_pitch = (raw_pitch * (1.0 - self.HPE_SMOOTHING)) + (self.smooth_pitch * self.HPE_SMOOTHING)
        else:
            ear, raw_yaw, raw_pitch = 0, 0, 0
            self.eye_closed_counter = 0
            self.smooth_yaw, self.smooth_pitch = 0, 0

        top_emotion_name = "N/A"
        top_emotion_score = 0.0
        if emotion_scores:
            top_emotion_name = max(emotion_scores, key=emotion_scores.get)
            top_emotion_score = emotion_scores[top_emotion_name]
            neutral_mood = emotion_scores.get('neutral', 0)
            positive_mood = max(emotion_scores.get('happiness', 0), emotion_scores.get('surprise', 0))
            negative_mood = max(
                emotion_scores.get('sadness', 0), 
                emotion_scores.get('anger', 0), 
                emotion_scores.get('fear', 0), 
                emotion_scores.get('disgust', 0)
            )
        else:
            positive_mood, negative_mood, neutral_mood = 0, 0, 0

        posture_score, posture_status = self._calculate_posture(pose_landmarks, frame_shape)
        good_mood = max(positive_mood, neutral_mood)
        bad_mood = negative_mood
        mood_score = (good_mood - bad_mood + 1) / 2 * 100
        
        yaw_focus = max(0, 1 - (abs(self.smooth_yaw) / self.HPE_YAW_THRESHOLD))
        pitch_focus = max(0, 1 - (abs(self.smooth_pitch) / self.HPE_PITCH_THRESHOLD))
        focus_score = min(yaw_focus, pitch_focus) * 100
        
        vigilance_score = 0 if (self.eye_closed_counter > 5) else 100
        
        if not face_landmarks:
            focus_score = 0
            vigilance_score = 0
        if posture_status in ["No Pose", "Partial Pose", "Error"]:
            posture_score = 50
        
        fused_score = (mood_score * 0.45) + \
                      (focus_score * 0.15) + \
                      (vigilance_score * 0.15) + \
                      (posture_score * 0.25)
        play_alert = False
        
        is_distracted = abs(self.smooth_yaw) > self.HPE_YAW_THRESHOLD or abs(self.smooth_pitch) > self.HPE_PITCH_THRESHOLD
        is_bad_posture = posture_status in ["Slouching", "Leaning"]
        
        if is_distracted:
            self.distracted_counter += 1
        else:
            self.distracted_counter = 0
            
        if is_bad_posture:
            self.bad_posture_counter += 1
        else:
            self.bad_posture_counter = 0

        if not face_landmarks:
            self.current_status = "No face detected"
            self.alert_status = "No Face"
        elif posture_status == "No Pose":
            self.current_status = "No pose detected"
            self.alert_status = "No Pose"
        elif self.eye_closed_counter > self.EAR_CONSEC_FRAMES:
            self.current_status = "Student is sleepy!"
            play_alert = True
            self.alert_status = "Sleepy_Alert"
            self.eye_closed_counter = 0 
        
        elif self.bad_posture_counter > self.POSTURE_CONSEC_FRAMES:
            self.current_status = f"Fix posture! ({posture_status})"
            play_alert = True
            self.alert_status = "Posture_Alert"
            self.bad_posture_counter = 0 
        
        elif self.distracted_counter > self.DISTRACTION_CONSEC_FRAMES:
            self.current_status = "Distracted!"
            play_alert = True
            self.alert_status = "Distracted_Alert"
            self.distracted_counter = 0 
        if play_alert and self.alert_status != "BREAK_Alert":
            self.alert_repeat_counter += 1
            
        if self.alert_repeat_counter > self.TAKE_A_BREAK_THRESHOLD:
            self.current_status = "Take a small break!"
            play_alert = True 
            self.alert_status = "BREAK_Alert"
            self.alert_repeat_counter = 0
        if not play_alert:
            if fused_score >= 80:
                self.current_status = "Engaged"
                self.alert_status = "None"
            elif fused_score >= 60:
                self.current_status = "Neutral"
                self.alert_status = "None"
            else: # Fused score is < 60
                self.current_status = "Needs to Focus"
                self.alert_status = "Low_Score"
        return {
            "status": self.current_status,
            "play_alert": play_alert,
            "fused_score": int(fused_score),
            "ear": ear,
            "yaw": self.smooth_yaw,
            "pitch": self.smooth_pitch,
            "top_emotion": top_emotion_name,
            "top_emotion_score": top_emotion_score,
            "posture_status": posture_status
        }