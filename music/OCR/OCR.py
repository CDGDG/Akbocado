import cv2
import numpy as np


# COLOR_BGR2RGB
def rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_chars(contour_list, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF,MIN_N_MATCHED, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, possible_contours):
    
    matched_result_idx = [] # possible_contours 에서 조건에 맞는 것들의 idx 값들을 담아서 리턴할 거시다.
    
    # contour_list 에서 2개의 contour (d1, d2) 의 '모든 조합'을 비교
    for d1 in contour_list:
        
        matched_contours_idx = []
        
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: continue # d1, d2 가 같은 contour 이면 비교할 필요 없다.
                
            # 두 contour 사이의 거리계산
            # d1 ~ d2 의 center 좌표 거리
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            # d1의 대각선 길이
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            
            # 각도 구하기 (아래 그림 참조)
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
                
            # 면적 비율, 폭 비율, 높이 비율
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']
            
            # 설정에 따른 기준에 맞는 contour 들만 골라서 
            # matched_contours_idx 에 contour 의 index 추가
            # 즉! d1 과 같은 번호판 후보군, d2(들)을 추가
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
        
        # 마지막에 d1도 넣어준다
        matched_contours_idx.append(d1['idx'])
        
        # matched_contours_idx 에 MIN_N_MATCHED 보다 개수가 적으면 번호판으로 인정안함
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue
        
        # 위 조건들을 다 통과하면 최종 후보군 matched_result_idx 에 인덱스 추가
        matched_result_idx.append(matched_contours_idx)
        
        # 그런데, 여기서 끝내지 않고
        # 최종 후보군에 들어있지 않은 것들을 한번 더 비교해 봅니다
        
        # 일단 matched_contours_idx 에 들지 않은 것들을 아래의 unmatched_contours_idx 에 넣고
        unmatched_contours_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contours_idx.append(d4['idx'])
                
        # np.take(a, idx)
        # a에서 idx 와 같은 인덱스의 값만 추출
        
        unmatched_contour =  np.take(possible_contours, unmatched_contours_idx)
        
        # 재귀호출
        recursive_contour_list = find_chars(unmatched_contour, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF,MIN_N_MATCHED, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, possible_contours)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        break
            
    return matched_result_idx