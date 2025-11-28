import cv2
import numpy as np
import math
#메인 실행 코드

#1. 카메라 연결 (0:외장 카메라, 2: 내장 카메라)
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.") 
        break 
   
    #연산 속도 향상을 위해 이미지 크기를 (640, 480)으로 고정
    frame = cv2.resize(frame, (640, 480))
    #frame.shape(높이, 가로 폭, 채널 수)
    #frame.shape[:2] : 처음 두 개의 요소만 높이, 가로폭만 할당
    height, width = frame.shape[:2]
    line_draw = np.copy(frame) 

    roi_y_start = height // 2  #높이의 절반부터
    roi_y_end = height 
    roi = frame[roi_y_start:roi_y_end, 0:width]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(roi_gray, (27, 27), 0)
    
    edges = cv2.Canny(blur, 20, 60)

    lines = cv2.HoughLinesP(edges, 
                            rho=1,              
                            theta=np.pi/180,    
                            threshold=40,       
                            minLineLength=20,   
                            maxLineGap=10) 
    
    #좌우 차선 분리 리스트 (왼쪽: 음수, 오른쪽: 양수)
    #y2-y1(음수)/x2-x1(양수) = 음수
    left_slopes = [] #왼쪽 기울기를 저장할 리스트 초기화
    #y2-y1(양수)/x2-x1(양수) = 양수
    right_slopes = [] #오른쪽 기울기를 저장할 리스트 초기화

    #절편 계산에 사용할 좌표 리스트
    left_coords = [] #모든 왼쪽 유효 선분의 좌표를 저장할 리스트 초기화
    right_coords = [] #모든 오른쪽 유효 선분의 좌표를 저장할 리스트 초기화
    
    #차선 분리를 위한 중앙 x 좌표 (640 / 2 = 320)
    center_x = width // 2 #기준선

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            #ROI 좌표를 전체 프레임 좌표로 변환(올바른 기울기 계산, 선분을 정확한 위치에 그림)
            #픽셀 크기 = 끝 좌표 - 시작 좌표 +1
            #y1(0부터 239) + roi_y_start(240) = y1_orig(240부터 479)
            y1_orig = y1 + roi_y_start 
            #y2(0부터 239) + roi_y_start(240) = y2_orig(240부터 479)
            y2_orig = y2 + roi_y_start

            if x2 - x1 == 0:
                slope = np.inf
                continue #0인 수직선은 무시하고 다음 선분으로 넘어감
                #print(f"좌표: ({x1}, {y1_orig}) -> ({x2}, {y2_orig}), 기울기: 수직선 (inf)")
            else:
                slope = (y2_orig - y1_orig) / (x2 - x1)
                #가로 방향에서 어디에 위치(중앙값)
                avg_x = (x1 + x2) // 2 #현재 처리 중인 선분의 가로 중앙값
                #print(f"좌표: ({x1}, {y1_orig}) -> ({x2}, {y2_orig}), 기울기: {slope:.2f}")

                #평균 기울기 계산을 위한 필터링 (기울기 10 미만)
                #10은 미니카 작동하면서 수정 10보다 작으면 직선 구간이 많거나 완만한 커브만 있는 길
                #10보다 크면 좁고 급커브가 많은 길
                if abs(slope) < 10: 
                    #좌우 분리
                    #기울기가 음수이고 평균x 좌표가 화면 중앙보다 작을때
                    if slope < 0 and avg_x < center_x:
                        #왼쪽 차선(음수 기울기, 화면 중앙 왼쪽)
                        left_slopes.append(slope) #유효하다고 판단된 왼쪽 기울기 값을 리스트 저장
                        left_coords.append([x1, y1_orig, x2, y2_orig]) #모든 왼쪽 유효 선분의 좌표를 리스트 저장
                        cv2.line(line_draw, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 0), 2)
                    #기울기가 양수이고 평균x 좌표가 화면 중앙보다 클때
                    elif 0 < slope and center_x < avg_x:
                          #오른쪽 차선: (양수 기울기, 화면 중앙 오른쪽)
                          right_slopes.append(slope) #유효하다고 판단된 오른쪽 기울기 값을 리스트 저장
                          right_coords.append([x1, y1_orig, x2, y2_orig]) #모든 오른쪽 유효 선분의 좌표를 리스트 저장
                          cv2.line(line_draw, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 2)
        
        if left_slopes: 
            #평균 기울기 = sigma(유효한 기울기 값)/ 유효한 기울기 개수 계산
            avg_left_slope = math.fsum(left_slopes) / len(left_slopes)
            
            #모든 유효 선분의 X, Y 좌표를 모아 평균을 구함
            all_x = np.array([coord[i] for coord in left_coords for i in [0, 2]]) #X 좌표만 추출(리스트 컴프리헨션)
            all_y = np.array([coord[i] for coord in left_coords for i in [1, 3]]) #Y 좌표만 추출(리스트 컴프리헨션)
            avg_x = np.mean(all_x)
            avg_y = np.mean(all_y)

            #절편 b = y - mx 계산
            avg_left_intercept = avg_y - (avg_left_slope * avg_x)

            #선분 좌표 계산 (화면 하단부터 중앙까지)
            y1_line = height #화면 하단 (가까운 지점)
            #0.6보다 작으면 차선이 더 길게 그려져 먼 곳을 예측하나 불안정해질 수 있음
            #0.6보다 크면 차선이 더 짧게 그려져 안정적이나 예측 거리가 줄어듦
            y2_line = int(height * 0.6) #화면 중앙 위쪽 (먼 지점)
            
            #x = (y - b) / m 공식을 사용하여 X 좌표 계산(직선의 방정식)
            x1_line = int((y1_line - avg_left_intercept) / avg_left_slope)
            x2_line = int((y2_line - avg_left_intercept) / avg_left_slope)
            
            cv2.line(line_draw, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 0), 10)
            print(f"평균 왼쪽 기울기: {avg_left_slope:.2f}")
        else:
            avg_left_slope = None
            print("평균 기울기를 계산할 유효한 선분이 부족함")
        if right_slopes:
            avg_right_slope = math.fsum(right_slopes) / len(right_slopes)

            # 모든 유효 선분의 x, y 좌표를 모아 평균을 구함
            all_x = np.array([coord[i] for coord in right_coords for i in [0, 2]]) #x 좌표만 추출(리스트 컴프리헨션)
            all_y = np.array([coord[i] for coord in right_coords for i in [1, 3]]) #y 좌표만 추출(리스트 컴프리헨션)
            avg_x = np.mean(all_x)
            avg_y = np.mean(all_y)

            #절편 b = y - mx 계산
            avg_right_intercept = avg_y - (avg_right_slope * avg_x)
            
            #선분 좌표 계산(화면 하단부터 중앙까지)
            y1_line = height #y 좌표의 최대값 (화면 하단, 차량에 가장 가까운 지점)
            #0.6보다 작으면 차선이 더 길게 그려져 먼 곳을 예측하나 불안정해질 수 있음
            #0.6보다 크면 차선이 더 짧게 그려져 안정적이나 예측 거리가 줄어듦
            y2_line = int(height * 0.6) #Y 좌표의 중간값 (화면 중앙 위쪽, 차선이 끝나는 지점)
            
            #x = (y - b) / m 공식을 사용하여 X 좌표 계산(직선의 방정식)
            x1_line = int((y1_line - avg_right_intercept) / avg_right_slope)
            x2_line = int((y2_line - avg_right_intercept) / avg_right_slope)

            cv2.line(line_draw, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 10)
            print(f"평균 오른쪽 기울기: {avg_right_slope:.2f}")
        else:
            avg_right_slope = None
            print("오른쪽 차선 기울기를 계산할 유효한 선분이 부족함")
    
    cv2.imshow("Frame", frame)
    cv2.imshow("roi", roi)
    cv2.imshow("grayscale", roi_gray)
    cv2.imshow("gaussian blur", blur)
    cv2.imshow("canny edges", edges)

    if cv2.waitKey(1) == ord('q'):
        break
    
# 카메라 장치 해제 및 모든 창 닫기
cap.release() 
cv2.destroyAllWindows()
