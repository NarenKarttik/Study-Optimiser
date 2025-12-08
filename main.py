import cv2
import proj.utils as utils
from proj.analyzers import FaceMeshAnalyzer, PoseAnalyzer, FerAnalyzer
from proj.logic import StudyOptimizerLogic
import time 

FER_MODEL_PATH = "models/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt" 
WEBCAM_ID = 0

def main():
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {WEBCAM_ID}")
        return

    face_mesh_analyzer = FaceMeshAnalyzer()
    pose_analyzer = PoseAnalyzer()
    
    try:
        fer_analyzer = FerAnalyzer(model_path=FER_MODEL_PATH)
    except RuntimeError as e:
        print(e)
        return

    optimizer_logic = StudyOptimizerLogic()

    alert_message = ""
    alert_end_time = 0
    ALERT_DURATION_SEC = 3 

    frame_count = 0
    total_fps = 0
    total_fer_latency = 0 
    total_mp_latency = 0  
    avg_fps = 0
    avg_fer_latency = 0   
    avg_mp_latency = 0    
 
    print("Starting Study Optimizer... Press 'q' to quit.")

    while cap.isOpened():

        loop_start_time = time.perf_counter()
        
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        mp_start_time = time.perf_counter()
        
        face_landmarks = face_mesh_analyzer.process_frame(frame)
        pose_results = pose_analyzer.process_frame(frame)

        mp_latency_ms = (time.perf_counter() - mp_start_time) * 1000
        total_mp_latency += mp_latency_ms
        
        
        pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
        
        results = None
        
        pose_analyzer.draw_landmarks(frame, pose_results)
        
        is_alert_active = time.time() < alert_end_time

        if face_landmarks:
            face_crop = utils.get_face_crop(frame, face_landmarks)
            
            emotion_scores = None
            if face_crop is not None:
                preprocessed_face = utils.preprocess_fer(face_crop)

                fer_latency_start = time.perf_counter()
                emotion_scores = fer_analyzer.predict(preprocessed_face)
                fer_latency_ms = (time.perf_counter() - fer_latency_start) * 1000
                total_fer_latency += fer_latency_ms

            results = optimizer_logic.update_metrics(
                frame, 
                face_landmarks, 
                pose_landmarks, 
                emotion_scores
            )

            if results["play_alert"]:
                utils.play_alert() 
                alert_message = results["status"]
                alert_end_time = time.time() + ALERT_DURATION_SEC
                is_alert_active = True

            draw_overlay(frame, results, is_alert_active, alert_message, avg_fps, avg_fer_latency, avg_mp_latency)
        else:
            cv2.putText(frame, "No face detected", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_count += 1
        loop_end_time = time.perf_counter()
        total_fps += 1.0 / (loop_end_time - loop_start_time)

        if frame_count % 10 == 0:
            avg_fps = total_fps / 10
            avg_fer_latency = total_fer_latency / 10 
            avg_mp_latency = total_mp_latency / 10   
            total_fps = 0
            total_fer_latency = 0 
            total_mp_latency = 0  
        
        cv2.imshow('Study Optimizer', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_overlay(frame, results, is_alert_active, alert_message, fps, fer_latency, mp_latency):
    """Helper function to draw all the info on the screen."""

    if results:
        score = results['fused_score']
        status = results['status']
        
        if score > 75: color = (0, 255, 0)
        elif score > 50: color = (0, 255, 255)
        else: color = (0, 0, 255)

        cv2.rectangle(frame, (10, 10), (210, 40), (0,0,0), -1)
        cv2.rectangle(frame, (10, 10), (10 + score * 2, 40), color, -1)
        cv2.putText(frame, f"SCORE: {score}", (15, 33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {status}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        top_emotion = results['top_emotion']
        top_score = results['top_emotion_score']
        if top_score > 0.20:
            emotion_text = f"MOOD: {top_emotion.capitalize()} ({top_score:.1%})"
            cv2.putText(frame, emotion_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        posture_status = results['posture_status']
        posture_text = f"POSTURE: {posture_status}"
        cv2.putText(frame, posture_text, (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        debug_text_ear = f"EAR: {results['ear']:.2f}"
        debug_text_yaw = f"Yaw: {results['yaw']:.1f}"
        debug_text_pitch = f"Pitch: {results['pitch']:.1f}"
        
        cv2.putText(frame, debug_text_ear, (10, frame.shape[0] - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, debug_text_yaw, (10, frame.shape[0] - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, debug_text_pitch, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    fps_text = f"FPS: {fps:.1f}"
    fer_latency_text = f"FER Latency: {fer_latency:.0f} ms" 
    mp_latency_text = f"MP Latency: {mp_latency:.0f} ms"   
    
    cv2.putText(frame, fps_text, (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, fer_latency_text, (frame.shape[1] - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, mp_latency_text, (frame.shape[1] - 200, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if is_alert_active:
        if int(time.time() * 2) % 2 == 0:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            cv2.rectangle(frame, (center_x - 250, center_y - 60), (center_x + 250, center_y + 60), (0, 0, 255), -1)
            cv2.rectangle(frame, (center_x - 250, center_y - 60), (center_x + 250, center_y + 60), (0, 0, 0), 2)
            
            cv2.putText(frame, alert_message, (center_x - 230, center_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

if __name__ == "__main__":
    main()