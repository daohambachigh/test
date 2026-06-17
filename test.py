from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os

# Các hằng số cho nhận diện ngã
ASPECT_RATIO_THRESHOLD = 0.7
TORSO_ANGLE_THRESHOLD = 30

def send_alert():
    """Gửi cảnh báo (có thể mở rộng với email, SMS)"""
    print("\n" + "="*50)
    print("FALL ALERT!")
    print("Fall detected! Please check the person.")
    print("="*50 + "\n")

def save_result_image(image, original_path):
    """Hàm tự động lưu hình ảnh sau khi nhận diện xong"""
    os.makedirs("runs", exist_ok=True)
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join("runs", f"{name}_result{ext}")
    cv2.imwrite(save_path, image)
    print(f"Đã tự động lưu hình ảnh tại: {save_path}")

# 1. Mở hộp thoại để người dùng chọn (upload) hình ảnh hoặc video
root = tk.Tk()
root.withdraw() # Ẩn cửa sổ chính của tkinter

file_path = filedialog.askopenfilename(
    title="Chọn hình ảnh hoặc video để nhận diện Pose",
    filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mkv")]
)

if file_path:
    print(f"Đã chọn file: {file_path}")
    
    # 2. Khởi tạo mô hình YOLOv8-pose
    model = YOLO('yolov8n-pose.pt')
    
    # Kiểm tra xem file là ảnh hay video
    is_video = file_path.lower().endswith(('.mp4', '.avi', '.mkv'))
    
    if is_video:
        cap = cv2.VideoCapture(file_path)
        
        # Cấu hình lưu video
        os.makedirs("runs", exist_ok=True)
        base_filename = os.path.basename(file_path)
        name, _ = os.path.splitext(base_filename)
        save_video_path = os.path.join("runs", f"{name}_result.mp4")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps: # Nếu lấy fps lỗi hoặc trả về NaN
            fps = 30.0
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video_path, fourcc, fps, (frame_width, frame_height))
        
        fall_counter = 0
        FALL_THRESHOLD = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            
            if len(results[0].boxes) > 0 and len(results[0].keypoints) > 0:
                boxes = results[0].boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = boxes
                width, height = x2 - x1, y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                keypoints = results[0].keypoints.xy[0]
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                
                shoulder_center = (left_shoulder + right_shoulder) / 2
                hip_center = (left_hip + right_hip) / 2
                
                dx = shoulder_center[0] - hip_center[0]
                dy = shoulder_center[1] - hip_center[1]
                
                angle_rad = np.arctan2(dx, dy)
                torso_angle = abs(np.degrees(angle_rad))
                if torso_angle > 90:
                    torso_angle = 180 - torso_angle
                
                if aspect_ratio > ASPECT_RATIO_THRESHOLD and torso_angle > TORSO_ANGLE_THRESHOLD:
                    fall_counter += 1
                else:
                    fall_counter = max(0, fall_counter - 1)
                
                if fall_counter >= FALL_THRESHOLD:
                    color = (0, 0, 255)
                    text = "FALL DETECTED!"
                    send_alert()
                else:
                    color = (0, 255, 0)
                    text = f"NORMAL (Fall frames: {fall_counter})"
                
                annotated_frame = results[0].plot()
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(annotated_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                info_text = f"AR: {aspect_ratio:.2f} | Angle: {torso_angle:.1f}° | Counter: {fall_counter}"
                cv2.putText(annotated_frame, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow("Fall Detection System", annotated_frame)
                out.write(annotated_frame)
            else:
                cv2.putText(frame, "No person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Fall Detection System", frame)
                out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Hoàn tất xử lý video! Đã lưu tại: {save_video_path}")
        
    else:
        # Xử lý hình ảnh (chạy 1 frame duy nhất)
        print("Đang xử lý hình ảnh...")
        frame = cv2.imread(file_path)
        results = model(frame)
        
        if len(results[0].boxes) > 0 and len(results[0].keypoints) > 0:
            boxes = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = boxes
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            keypoints = results[0].keypoints.xy[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            
            angle_rad = np.arctan2(dx, dy)
            torso_angle = abs(np.degrees(angle_rad))
            if torso_angle > 90:
                torso_angle = 180 - torso_angle
            
            # Với hình ảnh đơn, không dùng counter
            if aspect_ratio > ASPECT_RATIO_THRESHOLD and torso_angle > TORSO_ANGLE_THRESHOLD:
                color = (0, 0, 255)
                text = "FALL DETECTED!"
                send_alert()
            else:
                color = (0, 255, 0)
                text = "NORMAL"
                
            annotated_frame = results[0].plot()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(annotated_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            info_text = f"AR: {aspect_ratio:.2f} | Angle: {torso_angle:.1f}°"
            cv2.putText(annotated_frame, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow("Fall Detection System", annotated_frame)
        else:
            cv2.putText(frame, "No person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Fall Detection System", frame)
            
        # Dừng chờ người dùng nhấn phím bất kỳ trên cửa sổ ảnh để đóng tắt
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Gọi hàm lưu ảnh (sử dụng annotated_frame nếu có người, ngược lại dùng frame)
        final_image = annotated_frame if 'annotated_frame' in locals() else frame
        save_result_image(final_image, file_path)
        
        print("Hoàn tất! Cửa sổ đã đóng.")

else:
    print("Bạn chưa chọn file.")          